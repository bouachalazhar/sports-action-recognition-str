# Spatio-temporal random walk representations for action recognition

Reference implementation of the method described in:

> L. Bouacha, T. Magnien, L. Wendling, C. Kurtz. *Recognition of action phases
> from spatio-temporal intermediate representations in amateur sports
> competition videos.*
> Computer Vision and Image Understanding, 268:104748, 2026.

The method builds a compact 2D image from a short video clip by sampling random
walks through the video volume. Each walk visits one pixel per frame; the
visited pixels form the rows of an **STR image** of shape `T x L x C`, which a
standard 2D classifier can process. This avoids 3D convolutions while
preserving motion information, and keeps inference cheap enough for amateur
sports footage.

## Pipeline

1. **Region of interest** (`str_action.roi`): green chromaticity isolates the
   playing field, Farneback optical flow isolates motion, and the intersection
   keeps players moving on the field.
2. **Random walks** (`str_action.walk`): 8-connected walks of fixed length are
   sampled with an active support mask. With `--roi-mode all`, every step of
   the walk is constrained to the ROI; a step with no valid neighbour
   self-loops on the current pixel. Neighbourhood shape, stride, temporal
   padding and pixel aggregation are configurable.
3. **Classification** (`str_action.models`): ResNet-18, MobileNet V4,
   EfficientNet-V2, ConvNeXt, Swin-V2 and ViT-B/16 are supported. The paper's
   efficiency analysis uses MobileNet V4.
4. **Clip decision** (`str_action.evaluate`): per-image softmax probabilities
   are averaged per clip, then Top-1 and macro-F1 are reported at clip level.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
git clone https://github.com/bouachalazhar/sports-action-recognition-str.git
cd sports-action-recognition-str
uv sync --extra dev
```

## Quick start

Build STR images from a labelled video tree. Put each source or match in its
own group directory so that the grouped split can keep it intact:

```text
videos/
  train/
    action/match_001/clip_001.mp4
    action/match_002/clip_002.mp4
    non_action/match_003/clip_003.mp4
  val/
    ...
```

```bash
uv run str-build --source videos/train --destination data/train --walks 1000
uv run str-build --source videos/val --destination data/val --walks 1000
```

This writes `data/train/<label>/<group>/<clip>/str/*.png`, together with the
ROI masks and the sampled points. Each PNG is the full `T x L x C` STR image,
resized to `--image-size` (224 by default), not a single-frame summary. Use
`--roi-mode none` for the no-ROI ablation; the default `--roi-mode all`
constrains every step of the walk to the ROI.

Train a classifier:

```bash
uv run str-train \
  --data-root data/train \
  --model mobilenet_v4 \
  --epochs 20 \
  --pretrained \
  --seed 17 \
  --output-dir runs/mobilenet_v4
```

Evaluate a checkpoint on a held-out tree:

```bash
uv run str-evaluate \
  --data-root data/val \
  --checkpoint runs/mobilenet_v4/best.pt \
  --output metrics.json
```

Classify a raw video:

```bash
uv run str-infer match.mp4 \
  --checkpoint runs/mobilenet_v4/best.pt \
  --output-dir runs/inference
```

## Protocol

The manuscript trains with seeds 17, 29 and 43 and splits by source group
(match), so that clips, re-encodings and fragments of the same match never
appear in two splits. `str-train` groups clips by their directory below the
label, which is why the `match_XXX` level is required for a faithful protocol.
A tree with only one group per label is rejected instead of silently leaking
that group into both partitions.

Checkpoints are selected on validation macro-F1. Reported scores are
clip-level: per-image accuracy overweights clips that produce many STR images,
so it is not used as a selection criterion.

Checkpoints are loaded through PyTorch's restricted unpickler. A checkpoint
saved by v1.0.0 contains the old argument namespace and needs
`--allow-legacy-checkpoint` (only for a trusted local file); v1.0.1
checkpoints load without that flag.

## Repository layout

```text
src/str_action/
  walk.py       random walk sampling and STR construction
  roi.py        field, player and motion masks
  dataset.py    STR dataset, label mapping, grouped split
  models.py     2D classifier factory
  train.py      supervised training
  evaluate.py   clip-level metrics
  infer.py      end-to-end inference
  build.py      dataset construction
  checkpoints.py checkpoint compatibility loader
tests/          unit and smoke tests
```

## Tests

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

The tests generate synthetic videos, so no external data is needed. CI runs
lint, format, unit tests and a wheel build.

## Data and weights

This repository contains code only. No match footage, no model checkpoint and
no pre-computed STR image is distributed, because the broadcast rights of the
source videos do not allow redistribution. Bring your own videos and train the
models locally.

## Licence

Apache-2.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{bouacha2026cviu,
  title   = {Recognition of Action Phases from Spatio-Temporal Intermediate
             Representations in Amateur Sports Competition Videos},
  author  = {Bouacha, Lazhar and Magnien, Thierry and Wendling, Laurent and
             Kurtz, Camille},
  journal = {Computer Vision and Image Understanding},
  volume  = {268},
  pages   = {104748},
  year    = {2026},
  doi     = {10.1016/j.cviu.2026.104748}
}
```

The DOI above is the article DOI. No software archive DOI has been assigned
yet; one will be added to `CITATION.cff` only after a Zenodo deposit.
