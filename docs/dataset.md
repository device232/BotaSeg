# Rapeseed dataset format

The released training code expects a processed root with five disjoint folds,
named `Area_1` to `Area_5`. A fold contains one directory per plant:

```text
DATA_ROOT/
├── Area_1/
│   └── Plant_001/
│       ├── coord.npy       # float array, shape (N, 3)
│       ├── color.npy       # numeric array, shape (N, 3)
│       ├── segment.npy     # integer array, shape (N,) or (N, 1)
│       └── instance.npy    # integer array, shape (N,) or (N, 1)
└── Area_2/ ... Area_5/
```

Semantic labels are `0=leaf`, `1=petiole`, and `2=stem`; `-1` denotes an
ignored point. Instance labels use `-1` for ignored/unassigned points.

The repository contains neither these arrays nor source photographs. Before
making a public dataset archive, confirm ownership, participant/site policy,
annotation rights, and the intended license. The archive should include a
README, `metadata.tsv`, fixed fold assignments, a checksum manifest, and a
versioned DOI.

Do not merge the five folds into a single static `train/validation/test`
directory. The paper uses a five-run protocol in which each run contains
three training folds, a separate validation fold for checkpoint and
post-processing selection, and a distinct test fold used once only for final
evaluation. The canonical mapping is versioned in
`splits/rapeseed_5fold_train_val_test.json`:

| Run | Train folds | Validation fold | Test fold |
| --- | --- | --- | --- |
| run_01 | Area_3, Area_4, Area_5 | Area_2 | Area_1 |
| run_02 | Area_1, Area_4, Area_5 | Area_3 | Area_2 |
| run_03 | Area_1, Area_2, Area_5 | Area_4 | Area_3 |
| run_04 | Area_1, Area_2, Area_3 | Area_5 | Area_4 |
| run_05 | Area_2, Area_3, Area_4 | Area_1 | Area_5 |

Run the validator before training:

```bash
python tools/prepare_data/verify_rapeseed_layout.py --data-root "$DATA_ROOT"
python tools/prepare_data/emit_fold_protocol.py
```
