"""
PSegNet-style simultaneous semantic and instance segmentation.

This is a Pointcept adaptation of PSegNet. It preserves the current dataset,
training, and evaluation recipe, samples fixed-size anchor points inside the
model, and implements the paper's core blocks: DNFEB, DGFFM, and AM.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

import pointops
from pointcept.models.builder import MODELS


class PointBatchNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        return self.norm(x)


class SharedMLP(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Linear(in_channels, out_channels, bias=False),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )


class DNFEBStage(nn.Module):
    """Double-neighborhood feature extraction stage."""

    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.k = k
        self.feature_coord = nn.Linear(in_channels, 3)
        self.edge_mlp = SharedMLP(in_channels * 2, out_channels)
        self.value_mlp = SharedMLP(out_channels + 10, out_channels)
        self.attention_mlp = nn.Sequential(
            nn.Linear(out_channels + 10, out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 2, 1),
        )

    @staticmethod
    def position_encoding(coord, reference_index):
        neighbor_coord = pointops.grouping(reference_index, coord, coord, with_xyz=False)
        center_coord = coord.unsqueeze(1).expand_as(neighbor_coord)
        diff = center_coord - neighbor_coord
        dist = torch.norm(diff, p=2, dim=-1, keepdim=True)
        return torch.cat([center_coord, neighbor_coord, diff, dist], dim=-1)

    def forward(self, coord, feat, offset):
        xyz_index, _ = pointops.knn_query(self.k, coord.float().contiguous(), offset)
        feat_coord = self.feature_coord(feat)
        feat_index, _ = pointops.knn_query(
            self.k, feat_coord.float().contiguous(), offset
        )

        neighbor_feat = pointops.grouping(feat_index, feat, coord, with_xyz=False)
        center_feat = feat.unsqueeze(1).expand_as(neighbor_feat)
        edge_feat = torch.cat([center_feat, center_feat - neighbor_feat], dim=-1)
        n, k, _ = edge_feat.shape
        high_feat = self.edge_mlp(edge_feat.reshape(n * k, -1)).reshape(n, k, -1)

        low_feat = self.position_encoding(coord, xyz_index)
        mix_feat = torch.cat([high_feat, low_feat], dim=-1)
        weights = torch.softmax(self.attention_mlp(mix_feat).squeeze(-1), dim=1)
        value = self.value_mlp(mix_feat.reshape(n * k, -1)).reshape(n, k, -1)
        return torch.sum(value * weights.unsqueeze(-1), dim=1)


class DNFEB(nn.Module):
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.stage1 = DNFEBStage(in_channels, out_channels, k=k)
        self.stage2 = DNFEBStage(out_channels, out_channels, k=k)
        self.stage3 = DNFEBStage(out_channels * 2, out_channels, k=k)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Linear(in_channels, out_channels, bias=False)
        )

    def forward(self, coord, feat, offset):
        y1 = self.stage1(coord, feat, offset)
        y2 = self.stage2(coord, y1, offset)
        y3 = self.stage3(coord, torch.cat([y1, y2], dim=1), offset)
        return F.relu(y2 + y3 + self.shortcut(feat), inplace=True)


class AttentionModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    @staticmethod
    def scene_ranges(offset):
        start = 0
        for end in offset.detach().cpu().tolist():
            yield start, end
            start = end

    def forward(self, feat, offset):
        feat = feat * torch.sigmoid(feat.mean(dim=1, keepdim=True))
        output = feat.clone()
        for start, end in self.scene_ranges(offset):
            scene_feat = feat[start:end]
            avg_pool = scene_feat.mean(dim=0)
            max_pool = scene_feat.max(dim=0).values
            weight = torch.sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool))
            output[start:end] = scene_feat * weight.unsqueeze(0)
        return output


@MODELS.register_module("PSegNet-v1m1")
class PSegNet(nn.Module):
    def __init__(
        self,
        in_channels=6,
        semantic_num_classes=3,
        semantic_ignore_index=-1,
        segment_ignore_index=(-1,),
        instance_ignore_index=-1,
        stem_channels=64,
        encoder_channels=(64, 128, 256, 256),
        fusion_channels=128,
        embedding_dim=5,
        k=(16, 16, 8, 8),
        anchor_points=4096,
        cluster_thresh=0.6,
        cluster_closed_points=300,
        cluster_propose_points=50,
        cluster_min_points=10,
        semantic_loss_weight=1.0,
        instance_loss_weight=1.0,
        dhl_loss_weight=0.001,
        delta_same=0.5,
        delta_diff=1.5,
        dhl_delta_semantic=1.0,
        dhl_delta_instance=2.0,
        dhl_sample_points=2048,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.semantic_num_classes = semantic_num_classes
        self.semantic_ignore_index = semantic_ignore_index
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.anchor_points = anchor_points
        self.cluster_thresh = cluster_thresh
        self.cluster_closed_points = cluster_closed_points
        self.cluster_propose_points = cluster_propose_points
        self.cluster_min_points = cluster_min_points
        self.semantic_loss_weight = semantic_loss_weight
        self.instance_loss_weight = instance_loss_weight
        self.dhl_loss_weight = dhl_loss_weight
        self.delta_same = delta_same
        self.delta_diff = delta_diff
        self.dhl_delta_semantic = dhl_delta_semantic
        self.dhl_delta_instance = dhl_delta_instance
        self.dhl_sample_points = dhl_sample_points

        if isinstance(k, int):
            k = tuple([k] * len(encoder_channels))
        assert len(k) == len(encoder_channels)

        self.stem = SharedMLP(in_channels, stem_channels)
        blocks = []
        last_channels = stem_channels
        for channels, neighbours in zip(encoder_channels, k):
            blocks.append(DNFEB(last_channels, channels, k=neighbours))
            last_channels = channels
        self.encoder = nn.ModuleList(blocks)

        self.coarse_decoder = nn.Sequential(
            SharedMLP(encoder_channels[-1], fusion_channels),
            SharedMLP(fusion_channels, fusion_channels),
        )
        self.fine_projections = nn.ModuleList(
            [nn.Linear(channels, fusion_channels, bias=False) for channels in encoder_channels]
        )
        self.fine_decoder = nn.Sequential(
            SharedMLP(fusion_channels * len(encoder_channels), fusion_channels),
            SharedMLP(fusion_channels, fusion_channels),
        )
        self.dgf_fusion = nn.Sequential(
            SharedMLP(fusion_channels * 3, fusion_channels),
            SharedMLP(fusion_channels, fusion_channels),
        )

        self.instance_flow = SharedMLP(fusion_channels, fusion_channels)
        self.semantic_flow = SharedMLP(fusion_channels, fusion_channels)
        self.instance_attention = AttentionModule(fusion_channels)
        self.semantic_attention = AttentionModule(fusion_channels)
        self.embedding_head = nn.Linear(fusion_channels, embedding_dim)
        self.semantic_head = nn.Linear(fusion_channels, semantic_num_classes)
        self.ce_criteria = nn.CrossEntropyLoss(ignore_index=semantic_ignore_index)

    @staticmethod
    def _scene_ranges(offset):
        start = 0
        for end in offset.detach().cpu().tolist():
            yield start, end
            start = end

    def _sample_anchor_index(self, offset, device):
        indices = []
        anchor_offset = []
        total = 0
        for start, end in self._scene_ranges(offset):
            count = end - start
            if count <= self.anchor_points:
                local = torch.arange(count, device=device)
            elif self.training:
                local = torch.randperm(count, device=device)[: self.anchor_points]
                local = torch.sort(local).values
            else:
                local = torch.linspace(
                    0, count - 1, steps=self.anchor_points, device=device
                ).long()
            indices.append(local + start)
            total += local.numel()
            anchor_offset.append(total)
        return torch.cat(indices, dim=0), torch.tensor(anchor_offset, device=device).long()

    def _propagate_anchor_features(
        self, coord, offset, anchor_coord, anchor_feat, anchor_offset
    ):
        reference_index, _ = pointops.knn_query(
            1,
            anchor_coord.float(),
            anchor_offset.int(),
            coord.float(),
            offset.int(),
        )
        return anchor_feat[reference_index.flatten().long()]

    def _forward_features(self, data_dict):
        coord = data_dict["coord"]
        offset = data_dict["offset"]
        feat = self.stem(data_dict["feat"].float())
        anchor_index, anchor_offset = self._sample_anchor_index(offset, coord.device)
        anchor_coord = coord[anchor_index]
        anchor_feat = feat[anchor_index]

        encoder_features = []
        for block in self.encoder:
            anchor_feat = block(anchor_coord, anchor_feat, anchor_offset)
            encoder_features.append(anchor_feat)

        coarse = self.coarse_decoder(encoder_features[-1])
        fine = torch.cat(
            [proj(feature) for proj, feature in zip(self.fine_projections, encoder_features)],
            dim=1,
        )
        fine = self.fine_decoder(fine)
        fdgf_anchor = self.dgf_fusion(torch.cat([coarse, fine, coarse + fine], dim=1))
        fdgf = self._propagate_anchor_features(
            coord, offset, anchor_coord, fdgf_anchor, anchor_offset
        )

        instance_feat = self.instance_attention(self.instance_flow(fdgf), offset)
        semantic_feat = self.semantic_attention(self.semantic_flow(fdgf), offset)
        embedding = self.embedding_head(instance_feat)
        seg_logits = self.semantic_head(semantic_feat)
        return seg_logits, embedding, fdgf

    def _discriminative_loss(self, embedding, instance, offset):
        losses_same = []
        centers = []
        losses_reg = []
        for start, end in self._scene_ranges(offset):
            scene_embedding = embedding[start:end]
            scene_instance = instance[start:end]
            valid_instances = torch.unique(
                scene_instance[scene_instance != self.instance_ignore_index]
            )
            scene_centers = []
            for instance_id in valid_instances:
                mask = scene_instance == instance_id
                if mask.sum() == 0:
                    continue
                points = scene_embedding[mask]
                center = points.mean(dim=0)
                scene_centers.append(center)
                dist = torch.norm(points - center, p=2, dim=1)
                losses_same.append(F.relu(dist - self.delta_same).pow(2).mean())
                losses_reg.append(torch.norm(center, p=2))
            if scene_centers:
                centers.append(torch.stack(scene_centers, dim=0))

        same_loss = (
            torch.stack(losses_same).mean()
            if losses_same
            else embedding.new_tensor(0.0)
        )
        reg_loss = (
            torch.stack(losses_reg).mean() if losses_reg else embedding.new_tensor(0.0)
        )

        diff_losses = []
        for scene_centers in centers:
            if scene_centers.shape[0] < 2:
                continue
            dist = torch.cdist(scene_centers, scene_centers, p=2)
            mask = ~torch.eye(dist.shape[0], dtype=torch.bool, device=dist.device)
            diff_losses.append(F.relu(2 * self.delta_diff - dist[mask]).pow(2).mean())
        diff_loss = (
            torch.stack(diff_losses).mean()
            if diff_losses
            else embedding.new_tensor(0.0)
        )
        return same_loss + diff_loss + 0.001 * reg_loss

    def _dhl_loss(self, fdgf, segment, instance, offset):
        if self.dhl_loss_weight <= 0:
            return fdgf.new_tensor(0.0)

        losses = []
        for start, end in self._scene_ranges(offset):
            count = end - start
            if count < 2:
                continue
            sample_count = min(count, self.dhl_sample_points)
            if self.training and count > sample_count:
                local = torch.randperm(count, device=fdgf.device)[:sample_count]
            else:
                local = torch.linspace(0, count - 1, steps=sample_count, device=fdgf.device).long()
            idx = local + start
            feat = F.normalize(fdgf[idx], dim=1)
            sem = segment[idx]
            inst = instance[idx]
            dist = torch.cdist(feat, feat, p=2)

            valid = sem != self.semantic_ignore_index
            pair_valid = valid[:, None] & valid[None, :]
            eye = torch.eye(sample_count, dtype=torch.bool, device=feat.device)
            pair_valid = pair_valid & ~eye
            if pair_valid.sum() == 0:
                continue

            same_sem = sem[:, None] == sem[None, :]
            same_inst = (inst[:, None] == inst[None, :]) & (
                inst[:, None] != self.instance_ignore_index
            )
            diff_sem = pair_valid & ~same_sem
            diff_inst = pair_valid & same_sem & ~same_inst
            same_inst = pair_valid & same_inst

            parts = []
            if same_inst.any():
                parts.append(dist[same_inst].mean())
            if diff_inst.any():
                parts.append(F.relu(self.dhl_delta_semantic - dist[diff_inst]).mean())
            if diff_sem.any():
                parts.append(F.relu(self.dhl_delta_instance - dist[diff_sem]).mean())
            if parts:
                losses.append(torch.stack(parts).mean())

        return torch.stack(losses).mean() if losses else fdgf.new_tensor(0.0)

    def _cluster_predictions(self, seg_logits, embedding):
        semantic_scores = F.softmax(seg_logits, dim=-1)
        segment_pred = semantic_scores.max(1)[1].detach().cpu().numpy()
        embedding_np = embedding.detach().cpu().numpy()
        scores_np = semantic_scores.detach().cpu().numpy()

        masks = []
        classes = []
        scores = []
        for class_id in range(self.semantic_num_classes):
            point_index = np.where(segment_pred == class_id)[0]
            if point_index.size < self.cluster_min_points:
                continue
            clustering = DBSCAN(
                eps=float(self.cluster_thresh),
                min_samples=max(1, int(self.cluster_min_points)),
            ).fit(embedding_np[point_index])
            for label in np.unique(clustering.labels_):
                if label < 0:
                    continue
                local_mask = clustering.labels_ == label
                if local_mask.sum() <= self.cluster_propose_points:
                    continue
                mask = torch.zeros(seg_logits.shape[0], dtype=torch.int)
                mask[torch.from_numpy(point_index[local_mask]).long()] = 1
                masks.append(mask)
                classes.append(class_id)
                scores.append(float(scores_np[point_index[local_mask], class_id].mean()))

        if masks:
            pred_masks = torch.stack(masks, dim=0)
            pred_classes = torch.tensor(classes, dtype=torch.long)
            pred_scores = torch.tensor(scores, dtype=torch.float)
        else:
            pred_masks = torch.zeros((0, seg_logits.shape[0]), dtype=torch.int)
            pred_classes = torch.zeros(0, dtype=torch.long)
            pred_scores = torch.zeros(0, dtype=torch.float)
        return pred_scores, pred_masks, pred_classes

    def forward(self, data_dict):
        seg_logits, embedding, fdgf = self._forward_features(data_dict)
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        offset = data_dict["offset"]

        valid_segment = segment != self.semantic_ignore_index
        if valid_segment.any():
            seg_loss = self.ce_criteria(seg_logits, segment)
        else:
            seg_loss = seg_logits.sum() * 0.0
        instance_loss = self._discriminative_loss(embedding, instance, offset)
        dhl_loss = self._dhl_loss(fdgf, segment, instance, offset)
        loss = (
            self.semantic_loss_weight * seg_loss
            + self.instance_loss_weight * instance_loss
            + self.dhl_loss_weight * dhl_loss
        )
        return_dict = dict(
            loss=loss,
            seg_loss=seg_loss,
            instance_loss=instance_loss,
            dhl_loss=dhl_loss,
        )
        if not self.training:
            pred_scores, pred_masks, pred_classes = self._cluster_predictions(
                seg_logits, embedding
            )
            return_dict.update(
                seg_logits=seg_logits,
                pred_scores=pred_scores,
                pred_masks=pred_masks,
                pred_classes=pred_classes,
            )
        return return_dict
