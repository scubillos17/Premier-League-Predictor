from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .database import application_data_dir
from .service import refresh


def configure_logging() -> Path:
    log_dir = application_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "predictor.log"
    logging.basicConfig(
        filename=path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Premier League Predictor")
    parser.add_argument("--refresh-only", action="store_true", help="Refresh predictions without opening the GUI")
    parser.add_argument("--force", action="store_true", help="Clear a stale refresh lock older than 40 minutes")
    parser.add_argument("--db", type=Path, help="Override the local SQLite database path")
    args = parser.parse_args(argv)
    configure_logging()
    if args.refresh_only:
        result = refresh(force=args.force, database_path=args.db, progress=lambda message: logging.info(message))
        logging.info("Refresh result: %s", result)
        return 0 if result.ok else 1
    from .gui import run_gui

    return run_gui(database_path=args.db)


if __name__ == "__main__":
    sys.exit(main())
