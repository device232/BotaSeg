"""
Ablation variants for MyMethod-v1m1.

These classes are intentionally kept in a separate module so the paper
experiments can register controlled variants through config custom_imports
without changing the original model implementation.
"""

import torch
import torch.nn as nn

from pointcept.models.builder import MODELS, build_model
from pointcept.models.mymethod.mymethod_v1m1 import (
    BNReLULinear,
    BotanicalTopologyAwareNeck,
    GeometryAwareFusion,
    MyMethod,
    SemanticGeometryGatingHead,
)


class FeatureConcatFusion(nn.Module):
    """Fuse PTv3 and SpUNet features without coordinate-dependent gates."""

    def __init__(self, pt_channels=64, sp_channels=96, mid_channels=64):
        super().__init__()
        self.pt_project = BNReLULinear(pt_channels, mid_channels)
        self.sp_project = BNReLULinear(sp_channels, mid_channels)
        self.fusion_mlp = nn.Sequential(
            BNReLULinear(mid_channels * 2, mid_channels),
            BNReLULinear(mid_channels, mid_channels),
        )

    def forward(self, coord, pt_feat, sp_feat):
        del coord
        pt_feat = self.pt_project(pt_feat)
        sp_feat = self.sp_project(sp_feat)
        return self.fusion_mlp(torch.cat([pt_feat, sp_feat], dim=1))


class SemanticAuxOnlyNeck(nn.Module):
    """Keep auxiliary semantic routing but remove topology prototype reasoning."""

    def __init__(self, channels=64, num_classes=3, aux_hidden_channels=64):
        super().__init__()
        self.aux_head = nn.Sequential(
            BNReLULinear(channels, aux_hidden_channels),
            nn.Linear(aux_hidden_channels, num_classes),
        )

    def forward(self, feat):
        return feat, self.aux_head(feat)


class SharedOffsetHead(nn.Module):
    """Single shared offset regressor used for the w/o SGG-Head variant."""

    def __init__(self, channels=64):
        super().__init__()
        self.offset_head = nn.Sequential(
            BNReLULinear(channels, channels),
            BNReLULinear(channels, channels),
            nn.Linear(channels, 3),
        )

    def forward(self, feat, aux_logits):
        del aux_logits
        return self.offset_head(feat), None, None


def projection_or_identity(in_channels, out_channels):
    if in_channels == out_channels:
        return nn.Identity()
    return BNReLULinear(in_channels, out_channels)


@MODELS.register_module("MyMethod-v1m1-WoGAF")
class MyMethodWoGAF(MyMethod):
    """Dual-stream model with plain concat fusion instead of geometry-aware gates."""

    def __init__(self, *args, pt_channels=64, sp_channels=96, fusion_channels=64, **kwargs):
        super().__init__(
            *args,
            pt_channels=pt_channels,
            sp_channels=sp_channels,
            fusion_channels=fusion_channels,
            **kwargs,
        )
        self.fusion = FeatureConcatFusion(pt_channels, sp_channels, fusion_channels)


@MODELS.register_module("MyMethod-v1m1-WoBTA")
class MyMethodWoBTA(MyMethod):
    """Dual-stream model without botanical topology prototype reasoning."""

    def __init__(self, *args, fusion_channels=64, semantic_num_classes=3, **kwargs):
        super().__init__(
            *args,
            fusion_channels=fusion_channels,
            semantic_num_classes=semantic_num_classes,
            **kwargs,
        )
        self.bta_neck = SemanticAuxOnlyNeck(fusion_channels, semantic_num_classes)


@MODELS.register_module("MyMethod-v1m1-WoSGG")
class MyMethodWoSGG(MyMethod):
    """Dual-stream model with a single shared offset head."""

    def __init__(self, *args, fusion_channels=64, **kwargs):
        super().__init__(*args, fusion_channels=fusion_channels, **kwargs)
        self.sgg_head = SharedOffsetHead(fusion_channels)

    def _offset_losses(self, coord, offset_pred, expert_offsets, segment, instance, centroid):
        del expert_offsets, segment
        mask = (instance != self.instance_ignore_index).float()
        offset_gt = centroid - coord

        l1_dist = torch.sum(torch.abs(offset_pred - offset_gt), dim=-1)
        offset_l1_loss = torch.sum(l1_dist * mask) / (torch.sum(mask) + 1e-8)

        pred_norm = offset_pred / (
            torch.norm(offset_pred, p=2, dim=1, keepdim=True) + 1e-8
        )
        gt_norm = offset_gt / (torch.norm(offset_gt, p=2, dim=1, keepdim=True) + 1e-8)
        cosine = -(pred_norm * gt_norm).sum(-1)
        offset_cosine_loss = torch.sum(cosine * mask) / (torch.sum(mask) + 1e-8)
        expert_loss = offset_pred.new_tensor(0.0)
        return offset_l1_loss, offset_cosine_loss, expert_loss


class _SingleStreamMyMethod(MyMethod):
    """Single-backbone variant retaining BTA-Neck and SGG-Head."""

    def __init__(
        self,
        backbone,
        backbone_out_channels=64,
        fusion_channels=64,
        semantic_num_classes=3,
        semantic_ignore_index=-1,
        segment_ignore_index=(-1,),
        instance_ignore_index=-1,
        cluster_thresh=2.5,
        cluster_closed_points=300,
        cluster_propose_points=50,
        cluster_min_points=10,
        voxel_size=0.005,
        semantic_loss_weight=1.0,
        aux_loss_weight=0.4,
        offset_l1_loss_weight=1.0,
        offset_cosine_loss_weight=1.0,
        expert_offset_loss_weight=0.1,
    ):
        nn.Module.__init__(self)
        self.semantic_num_classes = semantic_num_classes
        self.semantic_ignore_index = semantic_ignore_index
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.cluster_thresh = cluster_thresh
        self.cluster_closed_points = cluster_closed_points
        self.cluster_propose_points = cluster_propose_points
        self.cluster_min_points = cluster_min_points
        self.voxel_size = voxel_size
        self.semantic_loss_weight = semantic_loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.offset_l1_loss_weight = offset_l1_loss_weight
        self.offset_cosine_loss_weight = offset_cosine_loss_weight
        self.expert_offset_loss_weight = expert_offset_loss_weight

        self.backbone = build_model(backbone)
        self.project = projection_or_identity(backbone_out_channels, fusion_channels)
        self.bta_neck = BotanicalTopologyAwareNeck(
            fusion_channels, semantic_num_classes
        )
        self.seg_head = nn.Linear(fusion_channels, semantic_num_classes)
        self.sgg_head = SemanticGeometryGatingHead(
            fusion_channels, semantic_num_classes
        )
        self.ce_criteria = nn.CrossEntropyLoss(ignore_index=semantic_ignore_index)

    def _forward_features(self, data_dict):
        feat = self.project(self.backbone(dict(data_dict)))
        coord = data_dict["coord"]
        common = min(coord.shape[0], feat.shape[0])
        coord = coord[:common]
        feat = feat[:common]
        bta_feat, aux_logits = self.bta_neck(feat)
        seg_logits = self.seg_head(bta_feat)
        offset_pred, expert_offsets, route = self.sgg_head(bta_feat, aux_logits)
        return coord, seg_logits, aux_logits, offset_pred, expert_offsets, route, common


@MODELS.register_module("MyMethod-v1m1-PTOnly")
class MyMethodPTOnly(_SingleStreamMyMethod):
    """PTv3-only backbone while retaining BTA-Neck and SGG-Head."""


@MODELS.register_module("MyMethod-v1m1-SpOnly")
class MyMethodSpOnly(_SingleStreamMyMethod):
    """SpUNet-only backbone while retaining BTA-Neck and SGG-Head."""

