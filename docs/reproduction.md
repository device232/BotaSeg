# Reproduction protocol

## Core five-fold experiment

The primary protocol is a five-run train-validation-test evaluation. In each
run, three folds are used for training, one distinct fold is used for
checkpoint selection and cluster-parameter calibration, and one remaining
fold is reserved for final testing. Neither checkpoint nor cluster parameters
may be selected on a test fold. The proposed configuration is
`configs/myno2paper/insseg-mymethod-v1m1-0-base.py`; the canonical mapping
is versioned in `splits/rapeseed_5fold_train_val_test.json`.

| Run | Train folds | Validation | Test |
| --- | --- | --- | --- |
| run_01 | Area_3, Area_4, Area_5 | Area_2 | Area_1 |
| run_02 | Area_1, Area_4, Area_5 | Area_3 | Area_2 |
| run_03 | Area_1, Area_2, Area_5 | Area_4 | Area_3 |
| run_04 | Area_1, Area_2, Area_3 | Area_5 | Area_4 |
| run_05 | Area_2, Area_3, Area_4 | Area_1 | Area_5 |

```bash
DATA_ROOT=/path/to/processed/data \
GPU=1 BATCH_SIZE=8 EPOCH=100 EVAL_EPOCH=100 \
bash scripts/train_rapeseed_5fold.sh
```

The configuration repeats the training dataset ten times per evaluation epoch,
matching the manuscript's 100 evaluation epochs with a 10-fold loader
repetition. Training writes one directory per test fold, named
`run_XX_test_Area_Y`. These outputs are not source-controlled.

After training, select clustering parameters on validation only and evaluate
the held-out test fold with the fixed selected values:

```bash
EXP_ROOT=exp/myno2paper/insseg-mymethod-v1m1-5fold \
bash scripts/postprocess_rapeseed_5fold.sh
```

Archive `validation_selection_fold_metrics.tsv`, `test_fold_metrics.tsv`,
`test_summary_metrics.tsv`, every validation sweep table, and the five copied
configs/checkpoints with the model release.

## Paper-to-code mapping

| Paper component | Configuration or tool |
| --- | --- |
| BotaSeg main result | `insseg-mymethod-v1m1-0-base.py` |
| PTv3 baseline | `insseg-pt-v3m1-0-base.py` |
| SpUNet, PointNet++, PlantNet, PSegNet baselines | Corresponding `insseg-*-v1m1-0-base.py` files |
| GAF/BTA/SGG ablations | `configs/myno2paper/ablations/` |
| Fold metric summaries | `tools/summarize_myno_5fold.py` |
| Cluster-parameter sweep | `tools/sweep_myno_cluster.py` |
| Semantic/instance point-cloud visualizations | `tools/visualize_myno_instances.py` |
| Organ traits and phenotype fingerprints | `tools/phenotype_demo/` |

## Required release artifacts before a manuscript claim is presented as fully reproducible

1. A versioned dataset archive and checksum manifest.
2. One checkpoint for each held-out fold, with its exact copied config.
3. The frozen metric and cluster-selection tables used in the manuscript.
4. A command manifest that maps every table and figure to source inputs.
5. The paper title, author list, DOI/preprint URL, and `CITATION.cff`.
