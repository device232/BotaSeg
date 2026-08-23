_base_ = ["insseg-pt-v3m1-0-base.py"]

# PointNet++ backbone for a controlled comparison: every non-model setting is
# inherited from the PTv3 experiment recipe.
model = dict(
    _delete_=True,
    type="PG-v1m1",
    backbone=dict(
        type="PointNet2-v1m1",
        in_channels=6,
        out_channels=64,
        channels=(32, 64, 128, 256, 512),
        strides=(1, 4, 4, 4, 4),
        nsamples=(8, 16, 16, 16, 16),
    ),
    backbone_out_channels=64,
    semantic_num_classes={{ _base_.num_classes }},
    semantic_ignore_index=-1,
    segment_ignore_index={{ _base_.segment_ignore_index }},
    instance_ignore_index=-1,
    cluster_thresh=1.5,
    cluster_closed_points=300,
    cluster_propose_points=50,
    cluster_min_points=10,
    voxel_size={{ _base_.grid_size }},
)
