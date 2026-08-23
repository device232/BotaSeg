_base_ = ["insseg-pt-v3m1-0-base.py"]

# Keep the full PTv3 experiment recipe and only replace the backbone model.
model = dict(
    _delete_=True,
    type="PG-v1m1",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=6,
        num_classes=0,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    backbone_out_channels=96,
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
