_base_ = ["../insseg-mymethod-v1m1-0-base.py"]

custom_imports = dict(
    imports=["pointcept.models.mymethod.mymethod_ablation_v1m1"],
    allow_failed_imports=False,
)

# SpUNet only: retain BTA-Neck and SGG-Head, remove the PTv3 stream.
model = dict(
    _delete_=True,
    type="MyMethod-v1m1-SpOnly",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=6,
        num_classes=0,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    backbone_out_channels=96,
    fusion_channels=64,
    semantic_num_classes={{ _base_.num_classes }},
    semantic_ignore_index=-1,
    segment_ignore_index={{ _base_.segment_ignore_index }},
    instance_ignore_index=-1,
    cluster_thresh=2.5,
    cluster_closed_points=300,
    cluster_propose_points=50,
    cluster_min_points=10,
    voxel_size={{ _base_.grid_size }},
    semantic_loss_weight=1.0,
    aux_loss_weight=0.4,
    offset_l1_loss_weight=1.0,
    offset_cosine_loss_weight=1.0,
    expert_offset_loss_weight=0.1,
)

