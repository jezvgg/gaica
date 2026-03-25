#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAICA local Python+Web series runner")
    parser.add_argument("--bot-a", required=True, help="path to bot A (.zip, dir with main.py, or .py)")
    parser.add_argument("--bot-b", required=True, help="path to bot B (.zip, dir with main.py, or .py)")
    parser.add_argument("--seed", type=int, default=1, help="match seed")
    parser.add_argument("--series-rounds", type=int, default=4, help="number of rounds/maps in one series")
    parser.add_argument("--round-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-cpu-seconds", type=float, default=120.0)
    parser.add_argument("--tick-response-timeout-seconds", type=float, default=1.0)
    parser.add_argument("--match-response-budget-seconds", type=float, default=60.0)
    parser.add_argument("--output", default="", help="output directory for artifacts")
    parser.add_argument("--print-outcome-json", action="store_true", help="print full outcome.json after run")
    return parser.parse_args()


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent]
    for candidate in candidates:
        if (candidate / "backend" / "runner" / "web_series_runner.py").exists() and (candidate / "game" / "web_port").exists():
            return candidate
    raise RuntimeError(
        "Unable to locate repository root. Expected backend/runner/web_series_runner.py and game/web_port."
    )


def _prepare_imports() -> None:
    repo = _repo_root()
    backend_dir = repo / "backend"
    game_dir = repo / "game"
    for candidate in (backend_dir, game_dir):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)


def _zip_from_directory(source_dir: Path, destination_zip: Path) -> None:
    with zipfile.ZipFile(destination_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_dir():
                continue
            relative = path.relative_to(source_dir)
            archive.write(path, arcname=str(relative))


def _normalize_bot_archive(bot_path: Path, temp_root: Path, slot: str) -> Path:
    if not bot_path.exists():
        raise FileNotFoundError(f"Bot path does not exist: {bot_path}")

    if bot_path.is_file() and bot_path.suffix.lower() == ".zip":
        return bot_path

    output_zip = temp_root / f"bot_{slot}.zip"
    if bot_path.is_file() and bot_path.suffix.lower() == ".py":
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(bot_path, arcname="main.py")
        return output_zip

    if bot_path.is_dir():
        main_py = bot_path / "main.py"
        if not main_py.exists() or not main_py.is_file():
            raise FileNotFoundError(f"Directory bot must contain main.py: {bot_path}")
        _zip_from_directory(bot_path, output_zip)
        return output_zip

    raise ValueError(f"Unsupported bot path format: {bot_path}")


def _default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return _repo_root() / "local-runner" / "dist" / "runs" / f"match_{stamp}"


def main() -> int:
    args = _parse_args()
    _prepare_imports()

    from runner.web_series_runner import run_series_match

    repo = _repo_root()
    os.environ.setdefault("GAICA_GAME_ROOT", str((repo / "game").resolve()))

    output_dir = Path(args.output).expanduser().resolve() if args.output else _default_output_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="gaica_local_runner_") as temp_dir:
        temp_root = Path(temp_dir)
        bot_a_zip = _normalize_bot_archive(Path(args.bot_a).expanduser().resolve(), temp_root, "a")
        bot_b_zip = _normalize_bot_archive(Path(args.bot_b).expanduser().resolve(), temp_root, "b")

        outcome = run_series_match(
            bot_a_zip=bot_a_zip,
            bot_b_zip=bot_b_zip,
            output_dir=output_dir,
            seed=int(args.seed),
            round_timeout_seconds=max(1.0, float(args.round_timeout_seconds)),
            max_cpu_seconds=max(1.0, float(args.max_cpu_seconds)),
            series_rounds=max(1, int(args.series_rounds)),
            match_id=f"local-{int(time.time())}",
            tick_response_timeout_seconds=max(0.01, float(args.tick_response_timeout_seconds)),
            match_response_budget_seconds=max(0.01, float(args.match_response_budget_seconds)),
        )

    print(f"outcome={output_dir / 'outcome.json'}")
    print(f"replay={output_dir / 'replay.json'}")
    print(f"log={output_dir / 'match.log'}")
    print(f"bot_a_stderr={output_dir / 'bot_a.stderr.log'}")
    print(f"bot_b_stderr={output_dir / 'bot_b.stderr.log'}")
    print(
        "summary="
        f"winner_slot={outcome.get('winner_slot')} "
        f"draw={outcome.get('draw')} "
        f"score={json.dumps(outcome.get('series_score', {}), ensure_ascii=False)} "
        f"ticks={outcome.get('ticks')}"
    )

    if args.print_outcome_json:
        print(json.dumps(outcome, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
