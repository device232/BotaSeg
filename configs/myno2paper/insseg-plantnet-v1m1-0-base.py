_base_ = ["insseg-pt-v3m1-0-base.py"]

# Pointcept adaptation of PlantNet. All non-model settings are inherited from
# the PTv3 experiment recipe for a controlled comparison.
model = dict(
    _delete_=True,
    type="PlantNet-v1m1",
    in_channels=6,
    semantic_num_classes={{ _base_.num_classes }},
    semantic_ignore_index=-1,
    segment_ignore_index={{ _base_.segment_ignore_index }},
    instance_ignore_index=-1,
    stem_channels=64,
    encoder_channels=(64, 96, 128, 128),
    fusion_channels=128,
    embedding_dim=5,
    k=16,
    anchor_points=4096,
    cluster_thresh=0.6,
    cluster_closed_points=300,
    cluster_propose_points=50,
    cluster_min_points=10,
    semantic_loss_weight=1.0,
    instance_loss_weight=1.0,
    fusion_loss_weight=0.001,
    delta_same=0.5,
    delta_diff=1.5,
    fusion_sample_points=2048,
)
