"""
PointNet++ backbone for Pointcept instance segmentation experiments.

This module implements the core PointNet++ set abstraction / feature
propagation design and returns per-point features for the PointGroup head.
"""

import torch
import torch.nn as nn
import pointops

from pointcept.models.builder import MODELS


class LinearBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        x = self.relu(self.bn(self.linear(x)))
        return x.reshape(*shape, -1)


class PointNetMLP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            layers.append(LinearBNReLU(in_channels, out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SetAbstraction(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, nsample=16):
        super().__init__()
        self.stride = stride
        self.nsample = nsample
        hidden_channels = max(out_channels // 2, 32)
        self.local_mlp = PointNetMLP(
            (in_channels + 3, hidden_channels, out_channels, out_channels)
        )

    @staticmethod
    def _downsample_offset(offset, stride):
        counts = offset.clone()
        counts[1:] = offset[1:] - offset[:-1]
        counts = torch.clamp(counts // stride, min=1)
        return torch.cumsum(counts, dim=0).int()

    def forward(self, point):
        coord, feat, offset = point
        coord = coord.float().contiguous()
        offset = offset.int().contiguous()

        if self.stride > 1:
            new_offset = self._downsample_offset(offset, self.stride).to(coord.device)
            idx = pointops.farthest_point_sampling(coord, offset, new_offset).long()
            new_coord = coord[idx, :]
        else:
            new_coord = coord
            new_offset = offset

        grouped_feat, _ = pointops.knn_query_and_group(
            feat,
            coord,
            offset=offset,
            new_xyz=new_coord,
            new_offset=new_offset,
            nsample=self.nsample,
            with_xyz=True,
        )
        feat = self.local_mlp(grouped_feat).max(dim=1)[0]
        return [new_coord, feat, new_offset]


class FeaturePropagation(nn.Module):
    def __init__(self, coarse_channels, skip_channels, out_channels):
        super().__init__()
        self.skip_channels = skip_channels
        self.mlp = PointNetMLP(
            (
                coarse_channels + skip_channels,
                out_channels,
                out_channels,
            )
        )

    def forward(self, fine_point, coarse_point):
        fine_coord, fine_feat, fine_offset = fine_point
        coarse_coord, coarse_feat, coarse_offset = coarse_point
        fine_coord = fine_coord.float().contiguous()
        coarse_coord = coarse_coord.float().contiguous()
        fine_offset = fine_offset.int().contiguous()
        coarse_offset = coarse_offset.int().contiguous()

        interp_feat = pointops.interpolation(
            coarse_coord,
            fine_coord,
            coarse_feat,
            coarse_offset,
            fine_offset,
        )
        feat = torch.cat([fine_feat, interp_feat], dim=1)
        feat = self.mlp(feat)
        return [fine_coord, feat, fine_offset]


@MODELS.register_module("PointNet2-v1m1")
class PointNet2(nn.Module):
    def __init__(
        self,
        in_channels=6,
        out_channels=64,
        channels=(32, 64, 128, 256, 512),
        strides=(1, 4, 4, 4, 4),
        nsamples=(8, 16, 16, 16, 16),
    ):
        super().__init__()
        if not (len(channels) == len(strides) == len(nsamples) == 5):
            raise ValueError("channels, strides, and nsamples must all have length 5")

        self.sa1 = SetAbstraction(in_channels, channels[0], strides[0], nsamples[0])
        self.sa2 = SetAbstraction(channels[0], channels[1], strides[1], nsamples[1])
        self.sa3 = SetAbstraction(channels[1], channels[2], strides[2], nsamples[2])
        self.sa4 = SetAbstraction(channels[2], channels[3], strides[3], nsamples[3])
        self.sa5 = SetAbstraction(channels[3], channels[4], strides[4], nsamples[4])

        self.fp4 = FeaturePropagation(channels[4], channels[3], channels[3])
        self.fp3 = FeaturePropagation(channels[3], channels[2], channels[2])
        self.fp2 = FeaturePropagation(channels[2], channels[1], channels[1])
        self.fp1 = FeaturePropagation(channels[1], channels[0], channels[0])
        self.out_mlp = PointNetMLP((channels[0], out_channels, out_channels))

    def forward(self, data_dict):
        point0 = [
            data_dict["coord"],
            data_dict["feat"],
            data_dict["offset"].int(),
        ]
        point1 = self.sa1(point0)
        point2 = self.sa2(point1)
        point3 = self.sa3(point2)
        point4 = self.sa4(point3)
        point5 = self.sa5(point4)

        point4 = self.fp4(point4, point5)
        point3 = self.fp3(point3, point4)
        point2 = self.fp2(point2, point3)
        point1 = self.fp1(point1, point2)
        return self.out_mlp(point1[1])
