# Changelog

## 1.1.1 — 2026-10-21

Patch release.

- Process the 1000 inference walks in bounded forward passes instead of
  materialising one 1000-image tensor. `str-infer` accepts `--batch-size`
  (default 32) and records the value in its report.
- Added a regression test comparing batched and single-pass probabilities.

## 1.1.0 — 2026-10-21

Protocol-fidelity release.

- Rebuilt the STR representation as the full `T x L x 3` image instead of a
  middle-frame summary.
- Applied the ROI support mask to every walk step and preserved the decoded
  temporal axis in the paper profile.
- Added a global source-group split, checkpoint class order checks, weighted
  cross-entropy, validation macro-F1 selection, early stopping and hard-vote
  clip aggregation.
- Added explicit YAML profiles (`configs/cviu_paper.yaml`) and checkpoint
  metadata for the representation and training protocol.
- Added safe checkpoint loading, an external-test workflow and regression tests
  for the representation, aggregation and configuration layers.
- Updated the README and protocol documentation.

## 1.0.2 — 2026-10-20

- Synchronised package metadata and release packaging.

## 1.0.1 — 2026-10-20

- Fixed STR fidelity and match-level split issues.

## 1.0.0 — 2026-10-19

- Initial public release.
