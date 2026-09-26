"""Small management CLI for ECG-Agent Framework."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .config import AgentConfig
from .registry import components
from .runtime import FRAMEWORK_VERSION


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ecg-framework")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("version", help="print the framework API version")
    validate = subparsers.add_parser(
        "validate-config", help="validate a framework YAML file"
    )
    validate.add_argument("path", type=Path)
    plugins = subparsers.add_parser("plugins", help="list registered plugins")
    plugins.add_argument("--discover", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "version":
        print(FRAMEWORK_VERSION)
    elif args.command == "validate-config":
        config = AgentConfig.from_yaml(args.path)
        print(json.dumps(config.as_dict(), indent=2, sort_keys=True))
    elif args.command == "plugins":
        loaded = components.discover() if args.discover else ()
        print(
            json.dumps(
                {"loaded_entry_points": loaded, "components": components.available()},
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
