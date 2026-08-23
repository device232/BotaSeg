"""
Hybrid PTv3-SpUNet segmentation network with botanical topology reasoning.

This is the Pointcept implementation of the method described in
docs/mymethod.pdf:
  - dual PTv3 / SpUNet backbone with cylindrical geometry-aware residual gates
  - BTA-Neck with semantic organ prototypes and fixed botanical adjacency
  - SGG-Head with semantic-routed class-specific offset experts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointgroup_ops import ballquery_batch_p, bfs_cluster
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils import offset2batch, batch2offset


class BNReLULinear(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )


class GeometryAwareFusion(nn.Module):
    def __init__(self, pt_channels=64, sp_channels=96, mid_channels=64):
        super().__init__()
        self.pt_project = BNReLULinear(pt_channels, mid_channels)
        self.sp_project = BNReLULinear(sp_channels, mid_channels)
        self.geo_norm = nn.BatchNorm1d(4)
        self.geo_mlp = nn.Sequential(
            BNReLULinear(4, mid_channels),
            nn.Linear(mid_channels, mid_channels * 2),
        )
        self.fusion_mlp = nn.Sequential(
            BNReLULinear(mid_channels * 2, mid_channels),
            BNReLULinear(mid_channels, mid_channels),
        )

    def forward(self, coord, pt_feat, sp_feat):
        pt_feat = self.pt_project(pt_feat)
        sp_feat = self.sp_project(sp_feat)

        radius = torch.sqrt(torch.sum(coord[:, :2].float().pow(2), dim=1, keepdim=True))
        geo = torch.cat([coord.float(), radius], dim=1)
        gates = self.geo_mlp(self.geo_norm(geo))
        pt_gate, sp_gate = gates.chunk(2, dim=1)
        pt_feat = (1.0 + torch.sigmoid(pt_gate)) * pt_feat
        sp_feat = (1.0 + torch.sigmoid(sp_gate)) * sp_feat
        return self.fusion_mlp(torch.cat([pt_feat, sp_feat], dim=1))


class BotanicalTopologyAwareNeck(nn.Module):
    def __init__(self, channels=64, num_classes=3, aux_hidden_channels=64):
        super().__init__()
        self.num_classes = num_classes
        self.aux_head = nn.Sequential(
            BNReLULinear(channels, aux_hidden_channels),
            nn.Linear(aux_hidden_channels, num_classes),
        )
        self.prototype_mlp = nn.Sequential(
            BNReLULinear(channels, channels),
            nn.Linear(channels, channels),
        )
        self.neck_fusion = nn.Sequential(
            BNReLULinear(channels * 2, channels),
            BNReLULinear(channels, channels),
        )

        adjacency = torch.eye(num_classes, dtype=torch.float)
        if num_classes == 3:
            adjacency = torch.tensor(
                [
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                dtype=torch.float,
            )
        adjacency = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.adjacency = nn.Parameter(adjacency)

    def forward(self, feat):
        aux_logits = self.aux_head(feat)
        prob = F.softmax(aux_logits, dim=1)
        route = prob.detach()
        point_norm = route / route.sum(dim=0, keepdim=True).clamp_min(1e-6)
        prototypes = torch.matmul(point_norm.transpose(0, 1), feat)
        prototypes = self.prototype_mlp(torch.matmul(self.adjacency, prototypes))
        topology_context = torch.matmul(route, prototypes)
        feat = self.neck_fusion(torch.cat([feat, topology_context], dim=1))
        return feat, aux_logits


class SemanticGeometryGatingHead(nn.Module):
    def __init__(self, channels=64, num_classes=3):
        super().__init__()
        self.geo_mlp = nn.Sequential(
            BNReLULinear(channels, channels),
            BNReLULinear(channels, channels),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    BNReLULinear(channels, channels),
                    nn.Linear(channels, 3),
                )
                for _ in range(num_classes)
            ]
        )

    def forward(self, feat, aux_logits):
        geo_feat = self.geo_mlp(feat)
        expert_offsets = torch.stack(
            [expert(geo_feat) for expert in self.experts], dim=1
        )
        route = F.softmax(aux_logits.detach(), dim=1)
        offset = torch.sum(expert_offsets * route.unsqueeze(-1), dim=1)
        return offset, expert_offsets, route


@MODELS.register_module("MyMethod-v1m1")
class MyMethod(nn.Module):
    def __init__(
        self,
        pt_backbone,
        sp_backbone,
        pt_channels=64,
        sp_channels=96,
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
        super().__init__()
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

        self.pt_backbone = build_model(pt_backbone)
        self.sp_backbone = build_model(sp_backbone)
        self.fusion = GeometryAwareFusion(pt_channels, sp_channels, fusion_channels)
        self.bta_neck = BotanicalTopologyAwareNeck(
            fusion_channels, semantic_num_classes
        )
        self.seg_head = nn.Linear(fusion_channels, semantic_num_classes)
        self.sgg_head = SemanticGeometryGatingHead(
            fusion_channels, semantic_num_classes
        )
        self.ce_criteria = nn.CrossEntropyLoss(ignore_index=semantic_ignore_index)

    @staticmethod
    def _align_features(coord, pt_feat, sp_feat):
        common = min(coord.shape[0], pt_feat.shape[0], sp_feat.shape[0])
        return coord[:common], pt_feat[:common], sp_feat[:common], common

    def _forward_features(self, data_dict):
        pt_feat = self.pt_backbone(dict(data_dict))
        sp_feat = self.sp_backbone(dict(data_dict))
        coord, pt_feat, sp_feat, common = self._align_features(
            data_dict["coord"], pt_feat, sp_feat
        )
        fused = self.fusion(coord, pt_feat, sp_feat)
        bta_feat, aux_logits = self.bta_neck(fused)
        seg_logits = self.seg_head(bta_feat)
        offset_pred, expert_offsets, route = self.sgg_head(bta_feat, aux_logits)
        return coord, seg_logits, aux_logits, offset_pred, expert_offsets, route, common

    def _offset_losses(self, coord, offset_pred, expert_offsets, segment, instance, centroid):
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
        if self.expert_offset_loss_weight > 0:
            parts = []
            for class_id in range(self.semantic_num_classes):
                class_mask = (
                    (segment == class_id)
                    & (instance != self.instance_ignore_index)
                    & (segment != self.semantic_ignore_index)
                )
                if class_mask.any():
                    parts.append(
                        F.smooth_l1_loss(
                            expert_offsets[class_mask, class_id, :],
                            offset_gt[class_mask],
                            reduction="mean",
                        )
                    )
            if parts:
                expert_loss = torch.stack(parts).mean()

        return offset_l1_loss, offset_cosine_loss, expert_loss

    def _cluster_predictions(self, coord, offset, seg_logits, offset_pred):
        center_pred = (coord + offset_pred) / self.voxel_size
        semantic_scores = F.softmax(seg_logits, dim=-1)
        segment_pred = torch.max(semantic_scores, 1)[1]

        ignore_masks = [
            (segment_pred == index).unsqueeze(-1)
            for index in self.segment_ignore_index
        ]
        if ignore_masks:
            valid_mask = ~torch.concat(ignore_masks, dim=1).sum(-1).bool()
        else:
            valid_mask = torch.ones_like(segment_pred, dtype=torch.bool)

        if valid_mask.sum() == 0:
            proposals_idx = torch.zeros(0).int()
            proposals_offset = torch.zeros(1).int()
        else:
            center_pred_ = center_pred[valid_mask]
            segment_pred_ = segment_pred[valid_mask]
            batch_ = offset2batch(offset)[valid_mask]
            offset_ = nn.ConstantPad1d((1, 0), 0)(batch2offset(batch_))
            idx, start_len = ballquery_batch_p(
                center_pred_,
                batch_.int(),
                offset_.int(),
                self.cluster_thresh,
                self.cluster_closed_points,
            )
            proposals_idx, proposals_offset = bfs_cluster(
                segment_pred_.int().cpu(),
                idx.cpu(),
                start_len.cpu(),
                self.cluster_min_points,
            )
            proposals_idx[:, 1] = (
                valid_mask.nonzero().view(-1)[proposals_idx[:, 1].long()].int()
            )

        proposals_pred = torch.zeros(
            (proposals_offset.shape[0] - 1, center_pred.shape[0]), dtype=torch.int
        )
        if proposals_idx.numel() > 0:
            proposals_pred[
                proposals_idx[:, 0].long(), proposals_idx[:, 1].long()
            ] = 1
        if proposals_offset.shape[0] > 1:
            instance_pred = segment_pred[
                proposals_idx[:, 1][proposals_offset[:-1].long()].long()
            ]
        else:
            instance_pred = segment_pred.new_zeros((0,))

        proposals_point_num = proposals_pred.sum(1)
        proposals_mask = proposals_point_num > self.cluster_propose_points
        proposals_pred = proposals_pred[proposals_mask]
        instance_pred = instance_pred[proposals_mask]

        pred_scores = []
        pred_classes = []
        pred_masks = proposals_pred.detach().cpu()
        for proposal_id in range(len(proposals_pred)):
            proposal = proposals_pred[proposal_id]
            object_class = instance_pred[proposal_id]
            confidence = semantic_scores[proposal.bool(), object_class].mean()
            pred_scores.append(confidence)
            pred_classes.append(object_class)

        if len(pred_scores) > 0:
            pred_scores = torch.stack(pred_scores).cpu()
            pred_classes = torch.stack(pred_classes).cpu()
        else:
            pred_scores = torch.tensor([])
            pred_classes = torch.tensor([])
        return pred_scores, pred_masks, pred_classes

    def forward(self, data_dict):
        (
            coord,
            seg_logits,
            aux_logits,
            offset_pred,
            expert_offsets,
            route,
            common,
        ) = self._forward_features(data_dict)

        segment = data_dict["segment"][:common]
        instance = data_dict["instance"][:common]
        centroid = data_dict["instance_centroid"][:common]
        offset = data_dict["offset"]

        valid_segment = segment != self.semantic_ignore_index
        if valid_segment.any():
            seg_loss = self.ce_criteria(seg_logits, segment)
            aux_loss = self.ce_criteria(aux_logits, segment)
        else:
            seg_loss = seg_logits.sum() * 0.0
            aux_loss = aux_logits.sum() * 0.0
        offset_l1_loss, offset_cosine_loss, expert_offset_loss = self._offset_losses(
            coord, offset_pred, expert_offsets, segment, instance, centroid
        )
        loss = (
            self.semantic_loss_weight * seg_loss
            + self.aux_loss_weight * aux_loss
            + self.offset_l1_loss_weight * offset_l1_loss
            + self.offset_cosine_loss_weight * offset_cosine_loss
            + self.expert_offset_loss_weight * expert_offset_loss
        )

        return_dict = dict(
            loss=loss,
            seg_loss=seg_loss,
            aux_loss=aux_loss,
            offset_l1_loss=offset_l1_loss,
            offset_cosine_loss=offset_cosine_loss,
            expert_offset_loss=expert_offset_loss,
        )
        if not self.training:
            pred_scores, pred_masks, pred_classes = self._cluster_predictions(
                coord, offset, seg_logits, offset_pred
            )
            return_dict.update(
                seg_logits=seg_logits,
                pred_scores=pred_scores,
                pred_masks=pred_masks,
                pred_classes=pred_classes,
            )
        return return_dict
