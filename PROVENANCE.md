# Provenance

This repository was assembled from the private `sportiq` research codebase at
commit `d1ec656e` (2026-10-19). The table below lists the correspondence between
the public modules and their private sources. The public code was cleaned of
local paths, cluster identifiers, project-specific settings and experiment
orchestration; the algorithms and their tests were kept.

| Public path | Private source |
| --- | --- |
| `src/str_action/walk.py` | `src/sportiq/features/random_walk.py` |
| `src/str_action/roi.py` | `src/sportiq/features/roi.py` |
| `src/str_action/dataset.py` | new: STR image dataset and grouped split |
| `src/str_action/models.py` | `src/sportiq/models/architectures/image_classifiers.py` (`NetSTR`), extended with a MobileNet V4 factory |
| `src/str_action/train.py` | new: supervised loop replacing the mock diagnostic script |
| `src/str_action/evaluate.py` | new: clip-level evaluation |
| `src/str_action/infer.py` | new: end-to-end inference |
| `src/str_action/build.py` | `src/sportiq/pipelines/data_processing/data_random_walk_make_data.py` |
| `tests/test_walk.py` | `tests/features/test_random_walk.py` |

The private repository remains the development source. This public repository
is the citable artefact associated with the article. No checkpoint, no match
footage and no pre-computed STR image is included.
