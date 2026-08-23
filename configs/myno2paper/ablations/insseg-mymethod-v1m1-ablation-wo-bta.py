_base_ = ["../insseg-mymethod-v1m1-0-base.py"]

custom_imports = dict(
    imports=["pointcept.models.mymethod.mymethod_ablation_v1m1"],
    allow_failed_imports=False,
)

# w/o BTA-Neck: keep the auxiliary semantic logits for SGG routing, but remove
# organ prototypes and leaf-petiole-stem topology reasoning.
model = dict(type="MyMethod-v1m1-WoBTA")

