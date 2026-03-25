#!/usr/bin/env python3
from __future__ import annotations

import sys

from gaica_bot.client import run_socket_bot
from gaica_bot.sample_bot import SampleBot


def main() -> int:
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 9001
    return run_socket_bot(host=host, port=port, bot=SampleBot())


if __name__ == "__main__":
    raise SystemExit(main())
