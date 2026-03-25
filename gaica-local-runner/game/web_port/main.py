from __future__ import annotations

import argparse
from pathlib import Path
import random
import subprocess
import sys
import time

from game.bot_tcp_server import BotTCPServer
from game.coordinator import MatchCoordinator
from game import config
from game.level_loader import get_levels_count, load_level
from game.simulation import GameSimulation
from game.web_server import WebServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web+Python port of hotline-miami-like with TCP bots")
    parser.add_argument("--bot-host", default="127.0.0.1")
    parser.add_argument("--bot-port", type=int, default=9001)
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--level-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round-time-limit", type=float, default=None)
    parser.add_argument("--with-test-bots", action="store_true")
    parser.add_argument(
        "--test-bot-a",
        choices=("aggressive", "defensive"),
        default="aggressive",
    )
    parser.add_argument(
        "--test-bot-b",
        choices=("aggressive", "defensive"),
        default="defensive",
    )
    parser.add_argument("--test-bot-b-seed", type=int, default=7)
    return parser.parse_args()


def _connect_host(bind_host: str) -> str:
    if bind_host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return bind_host


def _example_test_bot_script(mode: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    if mode == "aggressive":
        script = root / "examples" / "test_bots" / "aggressive" / "main.py"
    else:
        script = root / "examples" / "test_bots" / "defensive" / "main.py"
    if not script.exists():
        raise RuntimeError(f"Test bot script not found: {script}")
    return script


def _spawn_test_bot(mode: str, host: str, port: int, random_seed: int) -> subprocess.Popen[bytes]:
    del random_seed
    script = _example_test_bot_script(mode)
    cmd = [sys.executable, str(script), host, str(port)]
    return subprocess.Popen(cmd)


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    ldtk_path = root / "web_port" / "assets" / "levels" / "test_ldtk_project.ldtk"
    assets_dir = root / "web_port" / "assets"
    static_dir = root / "web_port" / "static"

    level_count = get_levels_count(ldtk_path)
    if level_count <= 0:
        raise RuntimeError("No levels found in LDtk project")

    rng = random.Random(args.seed)
    last_loaded_level_index: int | None = None
    rounds_built = 0
    series_level_indices: list[int] = []
    series_cursor = 0
    series_number = 0
    series_total_rounds = 1 if args.level_index is not None else level_count

    def _prepare_next_series() -> None:
        nonlocal series_level_indices, series_cursor, series_number
        series_number += 1
        if args.level_index is not None:
            series_level_indices = [args.level_index]
        else:
            series_level_indices = list(range(level_count))
            rng.shuffle(series_level_indices)
        series_cursor = 0
        if args.with_test_bots and series_number > 1 and args.level_index is None:
            print(f"Starting new full-map match #{series_number}")

    def _build_simulation_for_next_round() -> GameSimulation:
        nonlocal last_loaded_level_index, rounds_built, series_cursor

        if series_cursor >= len(series_level_indices):
            _prepare_next_series()

        selected_index = series_level_indices[series_cursor]
        map_in_series = series_cursor + 1
        series_cursor += 1

        level_obj = load_level(ldtk_path, level_index=selected_index)
        simulation_seed = rng.randrange(1_000_000_000)
        last_loaded_level_index = selected_index
        rounds_built += 1
        if args.with_test_bots and rounds_built > 1:
            print(
                f"Next map [{map_in_series}/{series_total_rounds}]: "
                f"{level_obj.identifier} (index={selected_index})"
            )
        return GameSimulation(
            level_obj,
            seed=simulation_seed,
            round_time_limit_seconds=args.round_time_limit,
        )

    simulation = _build_simulation_for_next_round()
    coordinator = MatchCoordinator(
        simulation,
        auto_restart_delay_seconds=(1.0 if args.with_test_bots else None),
        simulation_factory=(
            _build_simulation_for_next_round
            if series_total_rounds > 1 or args.with_test_bots
            else None
        ),
        series_total_rounds=series_total_rounds,
    )

    bot_server = BotTCPServer(args.bot_host, args.bot_port, coordinator)
    web_server = WebServer(args.web_host, args.web_port, coordinator, static_dir=static_dir, assets_dir=assets_dir)

    coordinator.start()
    bot_server.start()
    web_server.start()

    ui_host = _connect_host(args.web_host)

    print(f"Bot TCP server: {args.bot_host}:{args.bot_port}")
    print(f"Web UI: http://{ui_host}:{args.web_port}/")
    print(f"Tick rate: {config.TICK_RATE} tps")
    if series_total_rounds > 1:
        print(f"Series mode: one match = all maps ({series_total_rounds} total)")
        print(f"First map: {simulation.level.identifier} (index={last_loaded_level_index})")
    else:
        print(f"Selected level: {simulation.level.identifier} (index={last_loaded_level_index})")
    if args.with_test_bots:
        print("Auto rematch: enabled (1.0 sec after match end)")
    print("Waiting for 2 bots to connect...")

    bot_processes: list[subprocess.Popen[bytes]] = []

    if args.with_test_bots:
        bot_connect_host = _connect_host(args.bot_host)
        bot_processes.append(
            _spawn_test_bot(args.test_bot_a, bot_connect_host, args.bot_port, args.seed)
        )
        bot_processes.append(
            _spawn_test_bot(args.test_bot_b, bot_connect_host, args.bot_port, args.test_bot_b_seed)
        )
        print(f"Started test bots: {args.test_bot_a} vs {args.test_bot_b}")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for process in bot_processes:
            if process.poll() is None:
                process.terminate()
        for process in bot_processes:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()

        bot_server.stop()
        web_server.stop()
        coordinator.stop()


if __name__ == "__main__":
    main()
