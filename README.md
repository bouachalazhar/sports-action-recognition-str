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
   efficiency analysis uses MobileNet V4; the model is a profile choice, not a
   hard-coded part of the representation.
4. **Clip decision** (`str_action.evaluate`): every STR image votes for one
   class and the majority vote is the clip prediction. Softmax averaging is
   available as a diagnostic, but it is not the manuscript rule.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
git clone https://github.com/bouachalazhar/sports-action-recognition-str.git
cd sports-action-recognition-str
uv sync --extra dev
```

## Configuration-driven workflow

The general code is driven by YAML profiles. A file contains a shared
`representation` block and optional `build`, `train`, `evaluate` and
`inference` blocks:

```yaml
representation:
  walks: 1000
  walk_length: 224
  image_size: 224
  roi_mode: all
  temporal_policy: preserve
  save_points: false

train:
  batch_size: 256
  learning_rate: 0.0001
  epochs: 50
  class_weights: balanced
  clip_aggregation: vote
  validation_fraction: 0.1
  early_stopping_patience: 10
```

`configs/cviu_paper.yaml` is the explicit paper profile. `--config` supplies
defaults; an explicit command-line flag always wins. Unknown sections or
unknown command keys are rejected instead of being silently ignored. The
`representation` block is stored in the checkpoint and reused by `str-infer`
and `str-evaluate`, so inference cannot silently switch the walk length, image
size, ROI rule or temporal policy.

See [`configs/README.md`](configs/README.md) for the complete profile and the
list of values that are manuscript facts versus implementation choices.

## Quick start

Put each source video or match in its own group directory. The group is the
unit of the leak-free split:

```text
videos/
  train/
    action/match_001/clip_001.mp4
    action/match_002/clip_002.mp4
    non_action/match_003/clip_003.mp4
  test/
    ...
```

Build the training tree and the held-out test tree:

```bash
uv run str-build \
  --config configs/cviu_paper.yaml \
  --source videos/train --destination data/train

uv run str-build \
  --config configs/cviu_paper.yaml \
  --source videos/test --destination data/test
```

This writes `data/train/<label>/<group>/<clip>/str/*.png`, together with the
ROI masks. Each PNG is the full `T x L x C` STR image, resized to
`--image-size` (224 in the paper profile), not a single-frame summary. Sampled
walk coordinates are written only when `save_points: true` is set (or
`--save-points` is passed).

Train the classifier. `str-train` holds out source groups from the supplied
training tree for validation (`validation_fraction: 0.1`), while the separate
test tree is never seen during training:

```bash
uv run str-train \
  --config configs/cviu_paper.yaml \
  --data-root data/train \
  --model mobilenet_v4 \
  --pretrained \
  --output-dir runs/cviu_paper
```

Evaluate on the external test tree:

```bash
uv run str-evaluate \
  --config configs/cviu_paper.yaml \
  --data-root data/test \
  --checkpoint runs/cviu_paper/best.pt \
  --output metrics.json
```

Classify a raw video with the representation recorded in the checkpoint:

```bash
uv run str-infer match.mp4 \
  --checkpoint runs/cviu_paper/best.pt \
  --output-dir runs/inference
```

The default 1000 walks are processed in bounded forward passes
(`--batch-size 32`) so inference does not require a single 1000-image tensor.

## Protocol and manuscript alignment

| Item | Manuscript / paper profile | Implementation |
| --- | --- | --- |
| STR width `L` | 224 | `representation.walk_length` |
| Number of walks `N` | 1000 at inference | `representation.walks`; checkpoint metadata |
| Temporal axis `T` | decoded clip length, not artificially extended | `temporal_policy: preserve` |
| ROI | applied at every walk step | `roi_mode: all` |
| Image size | 224 | `representation.image_size` |
| Split | 80/10/10 by source group (match) | external 10% test tree + `validation_fraction: 0.1` on the remaining 90% |
| Optimiser | AdamW, `lr=1e-4` | `train.learning_rate` |
| Batch size | 256 | `train.batch_size` |
| Epochs | 50 with early stopping | `train.epochs`, `early_stopping_patience` |
| Class imbalance | class-weighted cross-entropy | `class_weights: balanced` |
| Clip aggregation | hard vote over STR images | `clip_aggregation: vote` |
| Selection | validation macro-F1 | checkpoint and history |
| Seeds | 17, 29, 43 | run the command three times with `--seed` |

A few values are implementation choices rather than claims from the article:

- `early_stopping_patience: 10` is the default patience used by this code; the
  article only specifies early stopping.
- `weight_decay: 1e-4` is the code default. It is recorded in the checkpoint
  and the training history, but it is not presented as a manuscript constant.
- `mobilenet_v4` in the paper profile is a template for the efficiency model.
  The same protocol can be run with another supported 2D classifier by passing
  `--model`.
- The three manuscript seeds are not averaged automatically. Run three
  independent commands and aggregate the resulting `metrics.json` files.

## Checkpoints and legacy behaviour

Checkpoints are selected on validation macro-F1 and store:

- the classifier state and class order;
- `representation`: walks, walk length, image size, ROI mode and temporal
  policy;
- `training`: class-weight mode, clip aggregation, early-stopping patience,
  seed, validation fraction and common optimisation hyper-parameters.

`str-infer` reuses the representation metadata unless a flag explicitly
overrides it. Checkpoints from v1.0.0 contain the old argument namespace and
require `--allow-legacy-checkpoint` (trusted local files only). v1.0.1 and
newer checkpoints load through PyTorch's restricted unpickler.

`temporal_policy: min-half-walk` is kept for reproducibility of the earlier
convenience behaviour: it pads very short clips to at least `L/2 + 1` frames.
With `L=224`, a 90-frame clip becomes 113 frames. It is **not** the manuscript
protocol; the paper profile uses `preserve`.

## Scope and limitations

- This repository contains code only. It does not redistribute match footage,
  pre-computed STR images, model weights or checkpoints. Bring your own videos.
- The repository does not claim to reproduce the numerical results of the
  article. Results depend on the private video corpus, the hardware, the
  random seeds and the exact source-group split.
- The test partition is an external tree. The repository cannot infer the
  original 80/10/10 partition from a flat video directory; create the held-out
  test tree before building STR images.
- `str-train` reports validation metrics from the groups held out of its input
  tree. A final test score must be produced by `str-evaluate` on the separate
  test tree.
- The `save_points` artefacts are diagnostic and disabled in the paper profile;
  they are not needed to train or evaluate.

## Repository layout

```text
src/str_action/
  config.py     YAML profile loading and defaults
  walk.py       random walk sampling and temporal policy
  roi.py        field, player and motion masks
  dataset.py    STR dataset, label mapping, grouped split
  models.py     2D classifier factory
  train.py      supervised training, class weights, early stopping
  evaluate.py   clip-level vote/softmax metrics
  infer.py      end-to-end inference from checkpoint metadata
  build.py      dataset construction
  checkpoints.py checkpoint compatibility loader
configs/        explicit protocols, including the CVIU paper profile
tests/          unit and smoke tests
```

## Tests

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv build
```

The tests generate synthetic videos and images, so no external data is needed.
CI runs lint, format, type-check, unit tests and a wheel build.

## Deterministic demo

`demo/` contains a self-contained synthetic fixture and a recorded transcript
of one complete build → train → evaluate run. Reproduce it with
`./demo/run_demo.sh`; see [`demo/TRANSCRIPT.md`](demo/TRANSCRIPT.md) for the
recorded outputs and the reproducibility scope.

## Release history

See [CHANGELOG.md](CHANGELOG.md).

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

```bibtex
@software{bouacha2026str-archive,
  title   = {Recognition of Action Phases from Spatio-Temporal Intermediate
             Representations in Amateur Sports Competition Videos},
  author  = {Bouacha, Lazhar and Magnien, Thierry and Wendling, Laurent and
             Kurtz, Camille},
  version = {1.2.0},
  year    = {2026},
  doi     = {10.5281/zenodo.23809003}
}
```

The `@article` entry identifies the journal paper. The `@software` entry
identifies the archived software version that accompanies it.
