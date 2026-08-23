# Recommended dataset release and linking plan

## Recommendation

Use **Zenodo as the canonical archival record** and **Hugging Face Datasets
as the downloadable mirror**. The processed rapeseed dataset currently has
36 plants in five fixed folds, 145 files, and a total size of about 2.2 GB.
It should not be committed to this GitHub code repository.

Zenodo is the citation target because it issues a DOI on publication and
supports versioned records. Cite the version-specific DOI in the manuscript;
the Zenodo concept DOI can point users to the latest version. Hugging Face is
the convenient distribution endpoint for users who want programmatic or
resumable downloads, but it is not the sole archival citation target.

## Archive contents

Create a versioned `v1.0.0` release containing:

```text
botaseg-rapeseed-3d-v1.0.0/
├── Area_1/ ... Area_5/             # coord/color/segment/instance .npy arrays
├── metadata.tsv                    # sample identifier, fold, phenotype group, and acquisition metadata
├── folds.json                      # fixed five-fold split used by the manuscript
├── README.md                       # data card and intended use
├── LICENSE-DATA                    # data-specific license
└── SHA256SUMS.txt                  # checksum for every distributed file
```

Publish the five `Area_*` folders as separate `.tar.zst` archives on Zenodo,
plus the metadata and checksum files. This keeps each download manageable
while preserving the exact directory layout expected by the code. Retain an
uncompressed directory mirror on Hugging Face so users can download only the
folds they need.

## Metadata required before publication

- Title, authors, affiliations, contact address, version, and publication date.
- Acquisition and reconstruction protocol; annotation definitions and quality control.
- Exact semantic mapping: `0=leaf`, `1=petiole`, `2=stem`, `-1=ignore`.
- Coordinate convention, units/scale, color encoding, and array shapes.
- Fold membership and a statement that folds are fixed for paper reproduction.
- License and any reuse restrictions; confirm the release is authorized by the
  data owner and field-site policy.
- Known limitations, intended research uses, and exclusions.

## Upload sequence

1. Create a Zenodo **Dataset** draft and reserve its DOI before publication.
   Upload the five archives, metadata, license, and `SHA256SUMS.txt`; then
   publish only after the code release and manuscript metadata are frozen.
2. Create a public Hugging Face Dataset repository, for example
   `device232/botaseg-rapeseed-3d`. Add a Dataset Card that repeats the data
   license, citation DOI, expected directory layout, and checksum location.
3. Upload the uncompressed data directory with the current Hugging Face CLI;
   it supports resumable folder uploads. Start with one fold as a test.

```bash
pip install -U huggingface_hub
hf auth login
HF_XET_HIGH_PERFORMANCE=1 hf upload device232/botaseg-rapeseed-3d \
  /absolute/path/to/myno2paperdatasetpreed . --repo-type dataset
```

4. Verify the downloaded files with `SHA256SUMS.txt`, then run:

```bash
python tools/prepare_data/verify_rapeseed_layout.py --data-root /path/to/data
```

5. Replace the pending rows in the repository README with both stable links:

```text
Dataset DOI: https://doi.org/10.xxxx/zenodo.xxxxxxx
Download mirror: https://huggingface.co/datasets/device232/botaseg-rapeseed-3d
```

## Versioning rule

Never replace files in a published Zenodo version. Any change to labels,
arrays, split membership, or metadata that affects reproducibility becomes a
new dataset version, with a new version DOI and an updated code release note.
