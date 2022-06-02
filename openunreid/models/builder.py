from einops import rearrange, repeat

import copy
import warnings

import torch
import torch.nn as nn
from torch.nn import init

from ..utils.dist_utils import convert_sync_bn, get_dist_info, simple_group_split
from ..utils.torch_utils import copy_state_dict, load_checkpoint
from .backbones import build_bakcbone
from .layers import build_embedding_layer, build_pooling_layer
from .utils.dsbn_utils import convert_dsbn

__all__ = ["ReIDBaseModel", "TeacherStudentNetwork",
            "build_model", "build_gan_model"]

class LAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.Nh   =  heads
        self.dk  =  dim
        self.dv  =  dim
        self.scale = dim ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_q.apply(weights_init_classifier)
        self.to_k.apply(weights_init_classifier)
        self.to_v.apply(weights_init_classifier)
    def forward(self, x,mask = None):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q = rearrange(q, 'b s n (h d) -> b s h n d', h=self.heads)
        k = rearrange(k, 'b s n (h d) -> b s h n d', h=self.heads)
        v = rearrange(v, 'b s n (h d) -> b s h n d', h=self.heads)
        dots = torch.einsum('bshid,bshjd->bshij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bshij,bshjd->bshid', attn, v)
        out = rearrange(out, 'b s h n d -> b s n (h d)')
        return out


class GAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.Nh   =  heads
        self.dk  =  dim
        self.dv  =  dim
        self.scale = dim ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_q.apply(weights_init_classifier)
        self.to_k.apply(weights_init_classifier)
        self.to_v.apply(weights_init_classifier)
    def forward(self, x,mask = None):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b  n (h d)')
        return out      


class Rlocalformer(nn.Module):
    def __init__(self, dim, heads,sh,sw,w):
        super().__init__()
        self.patch_size =1
        self.sw=sw
        self.sh=sh
        self.w=w
        self.attention =  LAttention(dim, heads=heads) 
        self.pre_ViT=nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=2,stride=1,padding="same")
        self.mix=nn.ConvTranspose2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1)
    def forward(self, x,mask = None):
        p = self.patch_size
        x_1 = self.pre_ViT(x)
        x_1 = rearrange(x_1, 'b c (sh h) (sw w) -> b (sh sw) c h w', sh=self.sh,sw=self.sw)
        x_1 = rearrange(x_1, 'b s c (h p1) (w p2) -> b s (h w) (p1 p2 c)', p1=p, p2=p)
        x_1 = self.attention(x_1,None)
        x_1 = rearrange(x_1, 'b (sh sw) (h w) (p1 p2 c) -> b c (sh h p1) (sw w p2)', w=self.w//self.sw,c=512,p1=p ,p2=p,sh=self.sh,sw=self.sw)
        x_1= self.mix(x_1)
        x = x_1+x
        return x

class Rglobalformer(nn.Module):
    def __init__(self, dim, heads,w):
        super().__init__()
        self.w=w
        self.attention =  GAttention(dim, heads=heads)
        self.conv=nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=2, stride=1,padding="same")
        self.mix= nn.ConvTranspose2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1)
    def forward(self, x,mask = None):
        p = 1
        z = self.conv(x)
        z = rearrange(z, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        z = self.attention(z,None)
        z = rearrange(z, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', w=self.w,c=512,p1=p ,p2=p)
        z = self.mix(z)
        x = z+x
        return x    
        

class ICMiF(nn.Module):
    def __init__(self, w=8,dim=512, heads=8):
        super().__init__()
        self.patch_size = 1  # 32
        self.transformer_1 = Rlocalformer(dim, heads,8,4,w)
        self.transformer_2 = Rglobalformer(dim, heads,w)
        self.transformer_3 = Rglobalformer(dim, heads,w)
        self.transformer_4 = Rlocalformer(dim, heads,8,4,w)
        self.transformer_5 = Rglobalformer(dim, heads,w)
        self.transformer_6 = Rglobalformer(dim, heads,w)
    def forward(self, x):
        x_1 = self.transformer_1(x)
        x_2 = self.transformer_2(x_1)
        x_3 = self.transformer_3(x_2)
        x_4 = self.transformer_4(x_3)
        x_5 = self.transformer_5(x_4)
        x_6 = self.transformer_6(x_5)
        return x_6


class ReIDBaseModel(nn.Module):
    """
    Base model for object re-ID, which contains
    + one backbone, e.g. ResNet50
    + one global pooling layer, e.g. avg pooling
    + one embedding block, (linear, bn, relu, dropout) or (bn, dropout)
    + one classifier
    """

    def __init__(
        self,
        arch,
        num_classes,
        pooling="avg",
        embed_feat=0,
        dropout=0.0,
        num_parts=0,
        include_global=True,
        pretrained=True,
    ):
        super(ReIDBaseModel, self).__init__()

        self.backbone=models.resnet50(pretrained=True)
        self.backbone_head = nn.Sequential(*list(self.backbone.children())[0:7])
        self.icmif=Transformer()

        self.global_pooling = build_pooling_layer(pooling)
        self.head = build_embedding_layer(
            1024, 0, 0
        )
        self.num_features = 1024

        self.num_classes = num_classes
        if self.num_classes > 0:
            self.classifier = nn.Linear(self.head.num_features, num_classes, bias=False)
            init.normal_(self.classifier.weight, std=0.001)

        self.num_parts = num_parts
        self.include_global = include_global

        if not pretrained:
            self.reset_params()

    @torch.no_grad()
    def initialize_centers(self, centers, labels):
        if self.num_classes > 0:
            self.classifier.weight.data[
                labels.min().item() : labels.max().item() + 1
            ].copy_(centers.to(self.classifier.weight.device))
        else:
            warnings.warn(
                f"there is no classifier in the {self.__class__.__name__}, "
                f"the initialization does not function"
            )

    def forward(self, x):
        batch_size = x.size(0)
        results = {}

        x = self.backbone_head(x)
        x = icmif(x)

        out = self.global_pooling(x)
        out = out.view(batch_size, -1)

        if self.num_parts > 1:
            assert x.size(2) % self.num_parts == 0
            x_split = torch.split(x, x.size(2) // self.num_parts, dim=2)
            outs = []
            if self.include_global:
                outs.append(out)
            for subx in x_split:
                outs.append(self.global_pooling(subx).view(batch_size, -1))
            out = outs

        results["pooling"] = out

        if isinstance(out, list):
            feat = [self.head(f) for f in out]
        else:
            feat = self.head(out)

        results["feat"] = feat

        if not self.training:
            return feat

        if self.num_classes > 0:
            if isinstance(feat, list):
                prob = [self.classifier(f) for f in feat]
            else:
                prob = self.classifier(feat)

            results["prob"] = prob

        return results

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class TeacherStudentNetwork(nn.Module):
    """
    TeacherStudentNetwork.
    """

    def __init__(
        self, net, alpha=0.999,
    ):
        super(TeacherStudentNetwork, self).__init__()
        self.net = net
        self.mean_net = copy.deepcopy(self.net)

        for param, param_m in zip(self.net.parameters(), self.mean_net.parameters()):
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        self.alpha = alpha

    def forward(self, x):
        if not self.training:
            return self.mean_net(x)

        results = self.net(x)

        with torch.no_grad():
            self._update_mean_net()  # update mean net
            results_m = self.mean_net(x)

        return results, results_m

    @torch.no_grad()
    def initialize_centers(self, centers, labels):
        self.net.initialize_centers(centers, labels)
        self.mean_net.initialize_centers(centers, labels)

    @torch.no_grad()
    def _update_mean_net(self):
        for param, param_m in zip(self.net.parameters(), self.mean_net.parameters()):
            param_m.data.mul_(self.alpha).add_(param.data, alpha=1-self.alpha)


def build_model(
    cfg, num_classes, init=None,
):
    """
    Build a (cross-domain) re-ID model
    with domain-specfic BNs (optional)
    """

    # construct a reid model
    model = ReIDBaseModel(
        cfg.MODEL.backbone,
        num_classes,
        cfg.MODEL.pooling,
        cfg.MODEL.embed_feat,
        cfg.MODEL.dropout,
        pretrained=cfg.MODEL.imagenet_pretrained,
    )

    # load source-domain pretrain (optional)
    if init is not None:
        state_dict = load_checkpoint(init)
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        copy_state_dict(state_dict, model, strip="module.")

    # convert to domain-specific bn (optional)
    num_domains = len(cfg.TRAIN.datasets.keys())
    if (num_domains > 1) and cfg.MODEL.dsbn:
        if cfg.TRAIN.val_dataset in list(cfg.TRAIN.datasets.keys()):
            target_domain_idx = list(cfg.TRAIN.datasets.keys()).index(
                cfg.TRAIN.val_dataset
            )
        else:
            target_domain_idx = -1
            warnings.warn(
                f"the domain of {cfg.TRAIN.val_dataset} for validation is not within "
                f"train sets, we use "
                f"{list(cfg.TRAIN.datasets.keys())[-1]}'s BN intead, "
                f"which may cause unsatisfied performance."
            )
        convert_dsbn(model, num_domains, target_domain_idx)
    else:
        if cfg.MODEL.dsbn:
            warnings.warn(
                "domain-specific BN is switched off, since there's only one domain."
            )
        cfg.MODEL.dsbn = False

    # create mean network (optional)
    if cfg.MODEL.mean_net:
        model = TeacherStudentNetwork(model, cfg.MODEL.alpha)

    # convert to sync bn (optional)
    rank, world_size, dist = get_dist_info()
    if cfg.MODEL.sync_bn and dist:
        if cfg.TRAIN.LOADER.samples_per_gpu < cfg.MODEL.samples_per_bn:

            total_batch_size = cfg.TRAIN.LOADER.samples_per_gpu * world_size
            if total_batch_size > cfg.MODEL.samples_per_bn:
                assert (
                    total_batch_size % cfg.MODEL.samples_per_bn == 0
                ), "Samples for sync_bn cannot be evenly divided."
                group_num = int(total_batch_size // cfg.MODEL.samples_per_bn)
                dist_groups = simple_group_split(world_size, rank, group_num)
            else:
                dist_groups = None
                warnings.warn(
                    f"'Dist_group' is switched off, since samples_per_bn "
                    f"({cfg.MODEL.samples_per_bn,}) is larger than or equal to "
                    f"total_batch_size ({total_batch_size})."
                )
            convert_sync_bn(model, dist_groups)

        else:
            warnings.warn(
                f"Sync BN is switched off, since samples ({cfg.MODEL.samples_per_bn, })"
                f" per BN are fewer than or same as samples "
                f"({cfg.TRAIN.LOADER.samples_per_gpu}) per GPU."
            )
            cfg.MODEL.sync_bn = False

    else:
        if cfg.MODEL.sync_bn and not dist:
            warnings.warn(
                "Sync BN is switched off, since the program is running without DDP"
            )
        cfg.MODEL.sync_bn = False

    return model


def build_gan_model(
    cfg,
    only_generator = False,
):
    """
    Build a domain-translation model
    """
    model = {}

    if only_generator:
        # for inference
        model['G'] = build_bakcbone(cfg.MODEL.generator)
    else:
        # construct generators
        model['G_A'] = build_bakcbone(cfg.MODEL.generator)
        model['G_B'] = build_bakcbone(cfg.MODEL.generator)
        # construct discriminators
        model['D_A'] = build_bakcbone(cfg.MODEL.discriminator)
        model['D_B'] = build_bakcbone(cfg.MODEL.discriminator)
        # construct a metric net for spgan
        if cfg.MODEL.spgan:
            model['Metric'] = build_bakcbone('metricnet')

    # convert to sync bn (optional)
    rank, world_size, dist = get_dist_info()
    if cfg.MODEL.sync_bn and dist:
        if cfg.TRAIN.LOADER.samples_per_gpu < cfg.MODEL.samples_per_bn:

            total_batch_size = cfg.TRAIN.LOADER.samples_per_gpu * world_size
            if total_batch_size > cfg.MODEL.samples_per_bn:
                assert (
                    total_batch_size % cfg.MODEL.samples_per_bn == 0
                ), "Samples for sync_bn cannot be evenly divided."
                group_num = int(total_batch_size // cfg.MODEL.samples_per_bn)
                dist_groups = simple_group_split(world_size, rank, group_num)
            else:
                dist_groups = None
                warnings.warn(
                    f"'Dist_group' is switched off, since samples_per_bn "
                    f"({cfg.MODEL.samples_per_bn,}) is larger than or equal to "
                    f"total_batch_size ({total_batch_size})."
                )

            for key in model.keys():
                convert_sync_bn(model[key], dist_groups)

        else:
            warnings.warn(
                f"Sync BN is switched off, since samples ({cfg.MODEL.samples_per_bn, })"
                f" per BN are fewer than or same as samples "
                f"({cfg.TRAIN.LOADER.samples_per_gpu}) per GPU."
            )
            cfg.MODEL.sync_bn = False

    else:
        if cfg.MODEL.sync_bn and not dist:
            warnings.warn(
                "Sync BN is switched off, since the program is running without DDP"
            )
        cfg.MODEL.sync_bn = False

    return model
