"""Interactive installer for the complete research dataset.

Each stage pulls its published Docker image and runs with the selected host
directory mounted at /data. Only Python's standard library is needed locally.
"""

import argparse
from datetime import date
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
from tempfile import TemporaryFile

from pumps_and_dumps.core.paths import (
    DATA_DIR_ENV,
    get_data_dir,
    get_root_dir,
    save_data_dir,
)
from installer.containers import container_stage, cleanup_container
from installer.state import Stage
from installer.tui import run_tui, tui_available
from installer.menu import Selection, choose_installation
from installer.plan import (
    STAGES,
    STAGE_KEYS,
    StageSettings,
    ordered_stages,
    validate_inputs,
    available_cpu_count,
)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Binance data, convert trades to Hive Parquet, then compute pump-event features.",
        epilog="Requires Python 3.10+ and Docker. Images are pulled before each stage. Without --data-dir, the installer asks where to store the data.",
    )
    parser.add_argument("--data-dir", help="Data directory to save and use (skips the prompt).")
    parser.add_argument(
        "--start-date",
        "--start",
        type=date.fromisoformat,
        help="Inclusive archive start (2018-01-01 unless a stage config specifies it).",
    )
    parser.add_argument(
        "--end-date",
        "--end",
        type=date.fromisoformat,
        help="Exclusive archive end (2025-01-01 unless a stage config specifies it).",
    )
    parser.add_argument(
        "--hive-workers",
        type=_positive_int,
        help="Parquet conversion processes (default: stage config or all available CPUs).",
    )
    parser.add_argument(
        "--feature-workers",
        type=_positive_int,
        help="Feature computation processes (default: stage config or all available CPUs).",
    )
    parser.add_argument(
        "--skip-existing-features",
        action="store_true",
        help="Reuse current-schema features; use only when their input data has not changed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview commands without prompting, saving configuration, or writing data.",
    )
    for stage in STAGE_KEYS:
        parser.add_argument(
            f"--{stage}-config",
            type=Path,
            help=f"HOCON/JSON config for {stage}; its directory is mounted read-only at /config. Use /data paths for output.",
        )
    display = parser.add_mutually_exclusive_group()
    display.add_argument(
        "--tui",
        action="store_true",
        help="Require the live terminal dashboard (automatic in a terminal).",
    )
    display.add_argument(
        "--no-tui",
        action="store_true",
        help="Use ordinary console output instead of the dashboard.",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--all",
        action="store_true",
        help="Run the full pipeline without the stage menu.",
    )
    scope.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_KEYS,
        help="Run only these stages, in pipeline order, without the stage menu.",
    )
    return parser


def _use_tui(parser: argparse.ArgumentParser, args: argparse.Namespace) -> bool:
    if args.no_tui or args.dry_run:
        return False
    if tui_available():
        return True
    if args.tui:
        parser.error("The TUI needs an interactive terminal with TERM set. Use --no-tui for redirected output.")
    return False


def _choose_selection(args: argparse.Namespace, use_tui: bool, dates: str) -> Selection | None:
    data_dir = get_data_dir() if args.data_dir is None else Path(args.data_dir).expanduser().resolve()
    if use_tui and not args.all and args.stages is None:
        configured = {key for key in STAGE_KEYS if getattr(args, f"{key}_config") is not None}
        return choose_installation(data_dir, dates, configured_stages=configured)

    if args.data_dir is None and not args.dry_run:
        answer = input(f"Where should the data be stored? [{data_dir}]: ").strip()
        if answer:
            data_dir = Path(answer).expanduser().resolve()
    return Selection(data_dir, ordered_stages(args.stages or STAGE_KEYS))


def _validate_selection(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    selection: Selection,
    root: Path,
) -> None:
    if not args.dry_run:
        # Configured stages validate their own possibly custom input locations.
        configured = {key for key in selection.stages if getattr(args, f"{key}_config") is not None}
        error = validate_inputs(selection.stages, selection.data_dir, skip_stages=configured)
        if error:
            raise ValueError(error)

    for key in selection.stages:
        config = getattr(args, f"{key}_config")
        if config is not None and not config.expanduser().is_file():
            parser.error(f"Config file not found: {config}")
        if key != "features" and config is None:
            start = args.start_date or date(2018, 1, 1)
            end = args.end_date or date(2025, 1, 1)
            if start >= end:
                parser.error("--end-date must be after --start-date (archive defaults: 2018-01-01 to 2025-01-01)")
    if not args.dry_run and shutil.which("docker") is None:
        raise ValueError("Docker is required. Install Docker and start its daemon before running the installer.")


def _build_stages(
    args: argparse.Namespace, root: Path, selected: list[str], data_dir: Path, *, progress: bool = False
) -> list[Stage]:
    settings = StageSettings(
        Path("/app"),
        args.start_date or date(2018, 1, 1),
        args.end_date or date(2025, 1, 1),
        args.hive_workers or available_cpu_count(),
        args.feature_workers or available_cpu_count(),
        args.skip_existing_features,
    )
    stages = []
    for option in STAGES:
        if option.key not in selected:
            continue
        config = getattr(args, f"{option.key}_config")
        options = option.options(settings)
        if config is not None:
            # The image's CLI reads config values. Only explicit host CLI flags
            # override them; installer defaults must not silently mask a config.
            config = config.expanduser().resolve()
            options = []
            if option.key != "features":
                if args.start_date is not None:
                    options.extend(["--start", str(args.start_date)])
                if args.end_date is not None:
                    options.extend(["--end", str(args.end_date)])
            if option.key == "parquet" and args.hive_workers is not None:
                options.extend(["--processes", str(args.hive_workers)])
            if option.key == "features":
                if args.feature_workers is not None:
                    options.extend(["--cpu-count", str(args.feature_workers)])
                if args.skip_existing_features:
                    options.append("--skip-existing")
        stages.append(
            container_stage(option.key, option.description, options, data_dir, config=config, progress=progress)
        )
    return stages


def _prepare_data_dir(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    # Check writability before saving the choice or starting a long download.
    with TemporaryFile(dir=data_dir):
        pass
    config_path = save_data_dir(data_dir)
    print(f"Saved data directory in {config_path}", flush=True)


def _run_console(stages: list[Stage], root: Path, environment: dict[str, str], dry_run: bool) -> int:
    for index, stage in enumerate(stages, start=1):
        print(f"\n[{index}/{len(stages)}] {stage.name}", flush=True)
        print(f"+ {shlex.join(stage.command)}", flush=True)
        if dry_run:
            continue
        try:
            subprocess.run(stage.command, cwd=root, env=environment, check=True)
        except subprocess.CalledProcessError as exc:
            print(
                f"\nFailed: {stage.name}. Later stages were not started.",
                file=sys.stderr,
            )
            return exc.returncode if exc.returncode > 0 else 128 - exc.returncode
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"\nInstallation failed during {stage.name}: {exc}", file=sys.stderr)
            return 1
        finally:
            cleanup_container(stage)
    return 0


def _print_completion(selection: Selection, dry_run: bool) -> None:
    if dry_run:
        print("\nPreview only; no configuration or data was changed.")
        return
    print(f"\nSelected stages complete. Data directory: {selection.data_dir}")
    if "features" in selection.stages:
        print(f"Feature files: {selection.data_dir / 'features' / 'pumps'}")
    print("Restart any running notebook kernel to use the saved data directory.")


def _install(parser: argparse.ArgumentParser, args: argparse.Namespace, use_tui: bool) -> int:
    root = get_root_dir()
    dates = f"{args.start_date or date(2018, 1, 1)} to {args.end_date or date(2025, 1, 1)} (exclusive)"
    if any(getattr(args, f"{key}_config") is not None for key in STAGE_KEYS if key != "features"):
        dates += "; configs may override defaults"
    selection = _choose_selection(args, use_tui, dates)
    if selection is None:
        print("Installation cancelled before starting any stages.")
        return 130
    _validate_selection(parser, args, selection, root)
    stages = _build_stages(args, root, selection.stages, selection.data_dir, progress=use_tui)
    print(f"Data directory: {selection.data_dir}", flush=True)
    print(
        f"Archive dates: {dates}",
        flush=True,
    )
    print(f"Selected stages: {', '.join(selection.stages)}", flush=True)
    if not args.dry_run:
        _prepare_data_dir(selection.data_dir)

    environment = {**os.environ, DATA_DIR_ENV: str(selection.data_dir)}
    if use_tui:
        code = run_tui(
            stages,
            root=root,
            environment=environment,
            data_dir=selection.data_dir,
            dates=dates,
        )
    else:
        code = _run_console(stages, root, environment, args.dry_run)
    if code:
        return code
    _print_completion(selection, args.dry_run)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.start_date is not None and args.end_date is not None and args.start_date >= args.end_date:
        parser.error("--end-date must be after --start-date")
    if args.data_dir is not None and not args.data_dir.strip():
        parser.error("--data-dir must be a nonempty path")
    use_tui = _use_tui(parser, args)

    previous_term = signal.signal(signal.SIGTERM, _interrupt)
    try:
        return _install(parser, args, use_tui)
    except EOFError:
        parser.error("No directory was supplied. Use --data-dir PATH for noninteractive runs.")
    except KeyboardInterrupt:
        print(
            "\nInstallation interrupted. Rerun the installer to reuse downloaded ZIPs.",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"\nInstallation failed during setup: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_term)


def _interrupt(_signum, _frame):
    """Let SIGTERM use the same container cleanup path as a keyboard interrupt."""
    raise KeyboardInterrupt


if __name__ == "__main__":
    raise SystemExit(main())
