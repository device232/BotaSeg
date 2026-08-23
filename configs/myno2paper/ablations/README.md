# MyMethod-v1m1 Ablation Configs

These configs are additive experiment definitions for the paper ablation table.
They do not modify the original `MyMethod-v1m1` implementation.

Variants:

- `insseg-mymethod-v1m1-ablation-full.py`: full model rerun.
- `insseg-mymethod-v1m1-ablation-wo-gaf.py`: remove geometry-aware gates.
- `insseg-mymethod-v1m1-ablation-pt-only.py`: PTv3 stream only, with BTA/SGG retained.(done)
- `insseg-mymethod-v1m1-ablation-sp-only.py`: SpUNet stream only, with BTA/SGG retained.(done)
- `insseg-mymethod-v1m1-ablation-wo-bta.py`: remove topology prototype reasoning. (done)
- `insseg-mymethod-v1m1-ablation-wo-sgg.py`: replace class-routed offset experts with one shared offset head. (done)

