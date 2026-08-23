# Pre-publication checklist

- [ ] Confirm the final repository name, owner, visibility, and default branch.
- [ ] Replace pending dataset/model archive entries in `README.md` with stable URLs and checksums.
- [ ] Confirm authorization and licensing for the 36-plant rapeseed dataset.
- [ ] Confirm that external benchmark datasets are linked, not redistributed.
- [ ] Add final author metadata, manuscript URL, and `CITATION.cff`.
- [ ] Audit every tracked file for local paths, credentials, personal data, raw images, logs, and checkpoints.
- [ ] Run the data validator on the public archive.
- [ ] Reproduce a one-fold evaluation in a clean environment.
- [ ] Run static checks and inspect `git status --ignored` before the first commit.
- [ ] Tag the release and archive it in Zenodo or an equivalent DOI provider.
