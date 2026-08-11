"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from dreamer_tools.adapters.dreamerv3 import discover
from dreamer_tools.bundle import inspect_bundle
from dreamer_tools.loader import check_load, evaluate, extract
from dreamer_tools.manifest import build_manifest, dump_manifest
from dreamer_tools.media import export_videos
from dreamer_tools.model import Discovery, Severity
from dreamer_tools.report import write_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    if args.command == "load-check":
        result = check_load(
            Path(args.path),
            Path(args.upstream),
            python=Path(args.python) if args.python else None,
            timeout=args.timeout,
        )
        print(("✓ " if result.ok else "✗ ") + result.message)
        if result.detail:
            print(result.detail)
        return 0 if result.ok else 2
    if args.command == "evaluate":
        result = evaluate(
            Path(args.path),
            Path(args.upstream),
            Path(args.output),
            steps=args.steps,
            envs=args.envs,
            platform=args.platform,
            python=Path(args.python) if args.python else None,
            timeout=args.timeout,
        )
        print(("✓ " if result.ok else "✗ ") + result.message)
        if result.detail:
            print(result.detail)
        return 0 if result.ok else 2
    if args.command == "bundle-info":
        try:
            info = inspect_bundle(Path(args.bundle))
        except (RuntimeError, ValueError) as exc:
            print(f"✗ {exc}")
            return 2
        print(json.dumps(info.metadata, indent=2, sort_keys=True))
        return 0
    if args.command == "extract":
        result = extract(
            Path(args.path),
            Path(args.upstream),
            Path(args.output),
            steps=args.steps,
            platform=args.platform,
            python=Path(args.python) if args.python else None,
            timeout=args.timeout,
        )
        print(("✓ " if result.ok else "✗ ") + result.message)
        if result.detail:
            print(result.detail)
        return 0 if result.ok else 2
    if args.command == "export-videos":
        try:
            outputs = export_videos(
                Path(args.bundle), Path(args.output_dir), fps=args.fps
            )
        except (RuntimeError, ValueError) as exc:
            print(f"✗ {exc}")
            return 2
        for output in outputs:
            print(f"✓ Wrote video: {output}")
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
    if args.command == "report":
        output = Path(args.output) if args.output else Path(args.path) / "report.html"
        try:
            write_report(
                run,
                output,
                bundle=Path(args.bundle) if args.bundle else None,
                videos=tuple(Path(value) for value in args.video),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"✗ Could not write report: {exc}")
            return 2
        print(f"Wrote report: {output}")
        _print_summary(run)
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
    load = subparsers.add_parser(
        "load-check", help="smoke-test agent restore with a pinned upstream checkout"
    )
    load.add_argument("path")
    load.add_argument(
        "--upstream", required=True, help="clean local DreamerV3 checkout"
    )
    load.add_argument("--python", help="Python executable with upstream dependencies")
    load.add_argument("--timeout", type=int, default=300, metavar="SECONDS")
    evaluation = subparsers.add_parser(
        "evaluate", help="run bounded evaluation-only execution upstream"
    )
    evaluation.add_argument("path")
    evaluation.add_argument("--upstream", required=True)
    evaluation.add_argument("--python")
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--steps", required=True, type=int)
    evaluation.add_argument("--envs", type=int, default=1)
    evaluation.add_argument("--platform", choices=("cpu", "cuda", "tpu"))
    evaluation.add_argument("--timeout", type=int, default=1800, metavar="SECONDS")
    bundle = subparsers.add_parser(
        "bundle-info", help="inspect a portable extraction bundle"
    )
    bundle.add_argument("bundle")
    videos = subparsers.add_parser(
        "export-videos", help="render bundle frame arrays as MP4 videos"
    )
    videos.add_argument("bundle")
    videos.add_argument("--output-dir", required=True)
    videos.add_argument("--fps", type=int, default=15)
    extraction = subparsers.add_parser(
        "extract", help="extract latents, logits, predictions, and dream frames"
    )
    extraction.add_argument("path")
    extraction.add_argument("--upstream", required=True)
    extraction.add_argument("--python")
    extraction.add_argument("--output", required=True)
    extraction.add_argument("--steps", required=True, type=int)
    extraction.add_argument("--platform", choices=("cpu", "cuda", "tpu"))
    extraction.add_argument("--timeout", type=int, default=1800, metavar="SECONDS")
    for name, help_text in (
        ("doctor", "diagnose a run directory"),
        ("manifest", "write a portable run manifest"),
        ("report", "write a self-contained HTML run report"),
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
        if name in ("manifest", "report"):
            command.add_argument("-o", "--output")
        if name == "report":
            command.add_argument("--bundle", help="extraction NPZ to summarize")
            command.add_argument(
                "--video",
                action="append",
                default=[],
                help="MP4 to embed; repeat for multiple videos",
            )
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
