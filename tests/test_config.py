"""Tests for the YAML protocol configuration layer."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from str_action.config import (
    REPRESENTATION_DEFAULTS,
    apply_defaults,
    load_config,
    representation,
    section,
)
from str_action.evaluate import parse_args as parse_evaluate_args
from str_action.train import parse_args

ROOT = Path(__file__).resolve().parents[1]


def test_cviu_profile_exposes_the_manuscript_protocol() -> None:
    config = load_config(ROOT / "configs" / "cviu_paper.yaml")
    protocol = representation(config)
    train = section(config, "train")

    assert protocol == {
        "walks": 1000,
        "walk_length": 224,
        "image_size": 224,
        "roi_mode": "all",
        "temporal_policy": "preserve",
        "save_points": False,
    }
    assert train["batch_size"] == 256
    assert train["learning_rate"] == pytest.approx(0.0001)
    assert train["epochs"] == 50
    assert train["validation_fraction"] == pytest.approx(0.1)
    assert train["class_weights"] == "balanced"
    assert train["clip_aggregation"] == "vote"


def test_representation_defaults_are_general_and_do_not_write_points() -> None:
    assert representation({}) == REPRESENTATION_DEFAULTS
    assert REPRESENTATION_DEFAULTS["save_points"] is False


def test_unknown_sections_and_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown configuration sections"):
        load_config_from_text("unknown: value\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    with pytest.raises(ValueError, match="unknown configuration keys"):
        apply_defaults(parser, {"epochs": 2, "typo": 3})


def test_known_only_ignores_keys_for_other_commands() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--walks", type=int, default=1)
    apply_defaults(parser, {"walks": 4, "save_points": True}, known_only=True)
    assert parser.parse_args([]).walks == 4


def test_cli_flags_override_yaml_defaults(tmp_path: Path) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(
        "representation:\n"
        "  walks: 7\n"
        "train:\n"
        "  epochs: 3\n"
        "  batch_size: 4\n"
        "  learning_rate: 0.001\n",
        encoding="utf-8",
    )
    args = parse_args(
        [
            "--config",
            str(path),
            "--data-root",
            str(tmp_path),
            "--epochs",
            "9",
        ]
    )
    assert args.epochs == 9
    assert args.batch_size == 4
    assert args.learning_rate == pytest.approx(0.001)
    assert args.walks == 7


def test_yaml_can_supply_required_train_values(tmp_path: Path) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(
        "train:\n  data_root: /tmp/data\n  epochs: 2\n",
        encoding="utf-8",
    )
    args = parse_args(["--config", str(path)])
    assert args.data_root == Path("/tmp/data")
    assert args.epochs == 2


def test_yaml_can_supply_required_evaluation_values(tmp_path: Path) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(
        "evaluate:\n"
        "  data_root: /tmp/data\n"
        "  checkpoint: /tmp/best.pt\n"
        "  batch_size: 7\n",
        encoding="utf-8",
    )
    args = parse_evaluate_args(["--config", str(path)])
    assert args.data_root == Path("/tmp/data")
    assert args.checkpoint == Path("/tmp/best.pt")
    assert args.batch_size == 7


def load_config_from_text(text: str) -> dict[str, object]:
    """Load a YAML snippet through a temporary file without extra fixtures."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(text)
        name = handle.name
    try:
        return load_config(Path(name))
    finally:
        Path(name).unlink()
