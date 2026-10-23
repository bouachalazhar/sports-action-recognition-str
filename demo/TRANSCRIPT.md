# Deterministic demo transcript

The demo proves that the packaged pipeline runs end to end and reproduces its
outputs inside the environment pinned by `uv.lock` (same platform, same
bundled codec build). The fixture uses closed-form content, so no random
number source is involved anywhere in the chain.

Fixture generation: `demo/generate_fixture.py` (12 training clips, 8 test
clips; 16 frames, 64 px, codec `mpeg4`, like the tests). Tree layout:

```
demo/data/train/<label>/<group>/clip_***.mp4
demo/data/test/<label>/<group>/clip_***.mp4
```

Reproduce with `./demo/run_demo.sh` from the repository root. Last verified
locally on a macOS arm CPU host:

```bash
uv run demo/generate_fixture.py
uv run str-build --config demo/demo_cviu_paper.yaml --source demo/data/train --destination demo/str/train
# ... converted 12 videos into demo/str/train
uv run str-build --config demo/demo_cviu_paper.yaml --source demo/data/test --destination demo/str/test
# ... converted 8 videos into demo/str/test
uv run str-train --config demo/demo_cviu_paper.yaml --data-root demo/str/train --output-dir demo/runs/str_demo
# {..., "epoch": 1, "train_loss": 0.7980956708391508, "val_top1": 0.5,
#  "val_macro_f1": 0.3333333333333333}
# {..., "epoch": 2, "train_loss": 0.5225889881451925, "val_top1": 0.5,
#  "val_macro_f1": 0.4857142857142857}
# best checkpoint: demo/runs/str_demo/best.pt
uv run str-evaluate --config demo/demo_cviu_paper.yaml --data-root demo/str/test --checkpoint demo/runs/str_demo/best.pt --output demo/metrics.json
# clips=8 top1=0.3750 macro_f1=0.3651
```

Determinism check: two consecutive executions of `./demo/run_demo.sh`
produced the identical evaluation file.

```
1bd20f064e3c96c7ee83c2460bcb0b7ec64d7ae61b16f19aa4799ccc5ff1c95a  demo/metrics.json
```

The recorded evaluation is byte-identical to the committed reference at
`demo/expected/metrics.json`; `shasum -a 256 demo/expected/metrics.json` gives
the same hash above.

Scope: `demo/runs/str_demo/history.json` is not a deterministic artefact (the
per-epoch wall clock is recorded on purpose); the STR demo uses `roi_mode:
none` and a tiny profile, so its numbers are demo numbers only and are not
paper results. On hosts outside the pinned `uv.lock` environment (different
platforms or unbundled codec builds), float rounding or codec differences may
change the last digits — the environment pin, not the demo data, provides the
reproducibility contract.
