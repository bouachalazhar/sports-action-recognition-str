# Protocol configurations

Each YAML file is a complete, readable protocol for the general implementation.
The command-line flags override the YAML defaults, which makes it possible to
reuse a profile for a quick ablation without editing the file.

| File | Purpose |
| --- | --- |
| `cviu_paper.yaml` | CVIU 2026 protocol: `L=224`, `N=1000`, preserved `T`, ROI at every step, 50 epochs, batch 256, `lr=1e-4`, weighted cross-entropy, hard-vote aggregation and a 90/10 source-group validation split inside the supplied training tree. |

## Sections

- `representation`: shared by `str-build`, `str-train` and `str-infer`. It is
  stored in the checkpoint.
  - `walks`: number of STR images per clip (`N`).
  - `walk_length`: number of visited pixels per walk (`L`).
  - `image_size`: resize applied to each STR image.
  - `roi_mode`: `all` applies the active ROI at every walk step; `none` uses
    the full frame.
  - `temporal_policy`: `preserve` keeps the decoded `T`; `min-half-walk` is a
    legacy padding policy and is not the paper protocol.
  - `save_points`: writes sampled coordinates to `points.json`; diagnostic and
    disabled by default.
- `build`: builder seed.
- `train`: model, optimiser, class weighting, early stopping and validation
  fraction.
- `evaluate`: evaluation batch size and aggregation. The checkpoint metadata is
  used when a value is omitted.
- `inference`: inference seed, the number of STR images per forward pass
  (`batch_size`) and optional overrides. Representation values are read from the
  checkpoint unless explicitly overridden.

## Validation split

The manuscript uses an 80/10/10 split by source group. The repository keeps the
test tree external:

1. hold out 10% of the source groups before building STR images;
2. pass the remaining 90% as `--data-root` to `str-train`;
3. `validation_fraction: 0.1` holds out 10% of those groups for validation,
   yielding the 80/10/10 partition;
4. evaluate the selected checkpoint with `str-evaluate` on the external tree.

Groups are directory names below the label. A match or source video must be a
single group so that clips from the same match cannot cross partitions.

## Values that are not manuscript constants

`early_stopping_patience`, `weight_decay`, the selected classifier and the
random seed of a single run are implementation choices. They are recorded in
the checkpoint and the training history rather than presented as constants of
the article.
