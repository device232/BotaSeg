# Reproduction protocol

## Core five-fold experiment

The primary protocol holds out one of `Area_1` to `Area_5` in turn. In each
run, the remaining four areas are used for training and the held-out area is
used for validation/testing. The proposed configuration is
`configs/myno2paper/insseg-mymethod-v1m1-0-base.py`.

```bash
DATA_ROOT=/path/to/processed/data \
GPU=1 BATCH_SIZE=8 EPOCH=1000 EVAL_EPOCH=100 \
bash scripts/train_rapeseed_5fold.sh
```

The script records fold-specific outputs in
`exp/myno2paper/insseg-mymethod-v1m1-5fold/val_Area_*`. These outputs are
not source-controlled. Freeze the generated `fold_metrics.tsv`,
`summary_metrics.tsv`, and clustering sweep tables in the model archive that
accompanies a paper release.

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
