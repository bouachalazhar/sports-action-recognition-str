#!/usr/bin/env zsh
# Reproduce the deterministic demo end to end and print SHA-256 of the
# recorded outputs. Run from the repository root with `uv` installed.
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
uv run demo/generate_fixture.py
uv run str-build --config demo/demo_cviu_paper.yaml --source demo/data/train --destination demo/str/train
uv run str-build --config demo/demo_cviu_paper.yaml --source demo/data/test --destination demo/str/test
uv run str-train --config demo/demo_cviu_paper.yaml --data-root demo/str/train --output-dir demo/runs/str_demo
uv run str-evaluate --config demo/demo_cviu_paper.yaml --data-root demo/str/test --checkpoint demo/runs/str_demo/best.pt --output demo/metrics.json

echo "--- SHA-256 of the deterministic output ---"
shasum -a 256 demo/metrics.json
