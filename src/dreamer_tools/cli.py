"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dreamer_tools.adapters.dreamerv3 import discover
from dreamer_tools.manifest import build_manifest, dump_manifest
from dreamer_tools.model import Discovery, Severity


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        versions = _parse_versions(args.environment_version)
    except ValueError as exc:
        parser.error(str(exc))
    run = discover(
        Path(args.path),
        dreamer_commit=args.dreamer_commit,
        model_preset=args.model_preset,
        python_version=args.python_version,
        environment_versions=versions,
    )
    if args.command == "doctor":
        _print_doctor(run)
        return int(run.severity)
    output = Path(args.output) if args.output else Path(args.path) / "dreamer-run.yaml"
    dump_manifest(build_manifest(run), output)
    print(f"Wrote manifest: {output}")
    _print_summary(run)
    return int(run.severity)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dreamer", description="Inspect trained DreamerV3 runs"
    )
    subparsers = parser.add_subparsers(dest="command")
    for name, help_text in (
        ("doctor", "diagnose a run directory"),
        ("manifest", "write a portable run manifest"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("path")
        command.add_argument(
            "--dreamer-commit", help="exact 40-character upstream Git commit"
        )
        command.add_argument(
            "--model-preset", help="training model preset, for example size50m"
        )
        command.add_argument(
            "--python-version", help="Python version used for training"
        )
        command.add_argument(
            "--environment-version",
            action="append",
            default=[],
            metavar="NAME=VERSION",
            help="repeatable environment package version",
        )
        if name == "manifest":
            command.add_argument("-o", "--output")
    return parser


def _parse_versions(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value or not all(value.split("=", 1)):
            raise ValueError(
                f"invalid environment version {value!r}; expected NAME=VERSION"
            )
        name, version = value.split("=", 1)
        result[name] = version
    return result


def _print_doctor(run: Discovery) -> None:
    symbols = {Severity.INFO: "✓", Severity.WARNING: "?", Severity.ERROR: "✗"}
    for item in run.diagnostics:
        print(f"{symbols[item.severity]} {item.message}")
    print()
    _print_summary(run)
    actions = [
        item.action
        for item in run.diagnostics
        if item.action and item.severity == run.severity
    ]
    if actions:
        print(f"Next action: {actions[0]}")


def _print_summary(run: Discovery) -> None:
    status = {
        Severity.INFO: "ready for structural inspection",
        Severity.WARNING: "incomplete provenance",
        Severity.ERROR: "run incomplete",
    }[run.severity]
    print(f"Status: {status}")


if __name__ == "__main__":
    sys.exit(main())
