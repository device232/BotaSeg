_base_ = ["../insseg-mymethod-v1m1-0-base.py"]

custom_imports = dict(
    imports=["pointcept.models.mymethod.mymethod_ablation_v1m1"],
    allow_failed_imports=False,
)

# w/o Geometry-Aware Fusion: remove the coordinate/radius residual gates and
# use plain projected PTv3+SpUNet concatenation.
model = dict(type="MyMethod-v1m1-WoGAF")

