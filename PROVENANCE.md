# Provenance

This repository is a cleaned, standalone release of research code developed for
the associated article. It was extracted from an internal research codebase;
local filesystem paths, cluster identifiers, private dataset locations,
experiment orchestration and project-specific configuration were removed. The
algorithms, public APIs and tests were retained and adapted for this release.

## Relationship to the article

The public modules implement the pipeline described in the article:

- region-of-interest estimation from green chromaticity and optical flow;
- random walks with an active ROI support mask;
- full spatio-temporal STR images of shape `T x L x C`;
- grouped train/validation splits at the source or match level;
- 2D classifier training and clip-level evaluation.

The article's reported numbers depend on the private broadcast dataset and on
the exact preprocessing and training schedule used for the paper. This
repository is a reference implementation, not a bit-for-bit reproduction of
those experiments.

## What is not included

No match footage, model checkpoint, pre-computed STR image or private dataset
is distributed. Broadcast and platform rights do not allow redistribution.
Bring your own videos and train the models locally.
