_base_ = ["../insseg-mymethod-v1m1-0-base.py"]

custom_imports = dict(
    imports=["pointcept.models.mymethod.mymethod_ablation_v1m1"],
    allow_failed_imports=False,
)

# w/o SGG-Head: replace class-routed offset experts with one shared offset head.
model = dict(type="MyMethod-v1m1-WoSGG", expert_offset_loss_weight=0.0)

