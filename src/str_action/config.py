"""YAML protocol configuration shared by the command-line tools.

A configuration file has one ``representation`` block and one block per
command (``build``, ``train``, ``evaluate``, ``inference``). The
``representation`` block records how the STR images were generated, so that
training can persist it in the checkpoint and inference can reuse it instead of
guessing different defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode

CONFIG_SECTIONS = ("representation", "build", "train", "evaluate", "inference")
REPRESENTATION_DEFAULTS: dict[str, Any] = {
    "walks": 1000,
    "walk_length": 224,
    "image_size": 224,
    "roi_mode": "all",
    "temporal_policy": "preserve",
    "save_points": False,
}


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects repeated mapping keys."""

    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        keys: list[Any] = []
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in keys:
                raise ValueError(f"duplicate YAML key: {key!r}")
            keys.append(key)
        return super().construct_mapping(node, deep=deep)


def config_path(argv: list[str] | None) -> Path | None:
    """Find ``--config`` before normal parsing so it can supply required values."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    for index, argument in enumerate(arguments):
        if argument == "--config" and index + 1 < len(arguments):
            return Path(arguments[index + 1])
        if argument.startswith("--config="):
            return Path(argument.split("=", 1)[1])
    return None


def load_config(path: Path | None) -> dict[str, Any]:
    """Load a YAML configuration, returning an empty mapping when omitted."""
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.load(handle, Loader=_UniqueKeyLoader) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    unknown = set(payload) - set(CONFIG_SECTIONS)
    if unknown:
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
    for name, value in payload.items():
        if not isinstance(value, dict):
            raise ValueError(f"configuration section {name!r} must be a mapping")
    return payload


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one configuration section, or an empty mapping."""
    if name not in CONFIG_SECTIONS:
        raise ValueError(f"unknown configuration section: {name!r}")
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"configuration section {name!r} must be a mapping")
    return dict(value)


def representation(config: dict[str, Any]) -> dict[str, Any]:
    """Return the representation block with explicit defaults applied."""
    payload = dict(REPRESENTATION_DEFAULTS)
    payload.update(section(config, "representation"))
    return payload


def apply_defaults(
    parser: Any, defaults: dict[str, Any], *, known_only: bool = False
) -> None:
    """Apply configuration values as argparse defaults.

    Args:
        parser: An :class:`argparse.ArgumentParser`.
        defaults: Values keyed by destination name.
        known_only: When true, silently ignore keys that the parser does not
            expose. This is used for the shared representation block, which
            contains a few keys that only some commands consume.
    """
    valid = {action.dest for action in parser._actions}  # noqa: SLF001
    unknown = set(defaults) - valid
    if unknown and not known_only:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    filtered = {key: value for key, value in defaults.items() if key in valid}
    for action in parser._actions:  # noqa: SLF001
        if (
            action.dest in filtered
            and filtered[action.dest] is not None
            and action.required
        ):
            action.required = False
    parser.set_defaults(**filtered)


__all__ = [
    "CONFIG_SECTIONS",
    "REPRESENTATION_DEFAULTS",
    "apply_defaults",
    "config_path",
    "load_config",
    "representation",
    "section",
]
