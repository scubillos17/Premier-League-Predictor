from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from .database import Database, application_data_dir
from .domain import RefreshResult
from .forecasting import MODEL_VERSION, create_forecasts
from .rules import CostGuard, validate_probability
from .sources import FPL_BASE, SourceClient, build_current_catalog


@contextmanager
def refresh_lock(force: bool = False) -> Iterator[None]:
    path = application_data_dir() / "refresh.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if force and age > 40 * 60:
            path.unlink(missing_ok=True)
        else:
            raise RuntimeError("Another refresh is already running")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        yield
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        path.unlink(missing_ok=True)


def validate_bundle(catalog: dict, predictions: list[dict]) -> None:
    roster_codes = {row["player_code"] for row in catalog["roster"]}
    fixture_codes = {row["code"] for row in catalog["fixtures"]}
    if len(roster_codes) < 300:
        raise RuntimeError("Current roster is unexpectedly small")
    if not predictions:
        raise RuntimeError("Prediction output is empty")
    seen: set[tuple[str, str]] = set()
    for row in predictions:
        key = (row["player_code"], row["fixture_code"])
        if key in seen:
            raise RuntimeError(f"Duplicate prediction for {key}")
        seen.add(key)
        if row["player_code"] not in roster_codes:
            raise RuntimeError(f"Prediction contains a non-current player: {row['player_code']}")
        if row["fixture_code"] not in fixture_codes:
            raise RuntimeError(f"Prediction contains an unknown fixture: {row['fixture_code']}")
        for name in (
            "defensive_return_probability", "clean_sheet_probability", "yellow_probability",
            "red_probability", "start_probability", "bench_probability", "no_appearance_probability",
        ):
            if not validate_probability(float(row[name])):
                raise RuntimeError(f"Invalid {name} for player {row['player_code']}")
        total = row["start_probability"] + row["bench_probability"] + row["no_appearance_probability"]
        if abs(total - 1.0) > 0.00001:
            raise RuntimeError(f"Appearance probabilities do not sum to one for {row['player_code']}")


def refresh(
    force: bool = False,
    database_path: Path | str | None = None,
    progress: Callable[[str], None] | None = None,
) -> RefreshResult:
    CostGuard().assert_zero_cost()
    database = Database(database_path)
    audit_id = f"audit-{uuid.uuid4().hex}"
    database.begin_audit(audit_id)
    try:
        with refresh_lock(force):
            if progress:
                progress("Downloading the current Premier League roster")
            client = SourceClient()
            bootstrap_bytes = client.fetch(
                f"{FPL_BASE}/bootstrap-static/", "current FPL roster", allow_cached_on_error=False
            )
            fixtures_bytes = client.fetch(
                f"{FPL_BASE}/fixtures/", "current FPL fixtures", allow_cached_on_error=False
            )
            bootstrap = json.loads(bootstrap_bytes)
            fixtures_feed = json.loads(fixtures_bytes)
            catalog = build_current_catalog(bootstrap, fixtures_feed)
            predictions, observations = create_forecasts(catalog, client, progress)
            validate_bundle(catalog, predictions)
            now = datetime.now(timezone.utc)
            run_id = f"{now:%Y%m%dT%H%M%SZ}-gw{catalog['gameweek']}"
            snapshot_id = f"roster-{now:%Y%m%dT%H%M%SZ}"
            for row in predictions:
                row["run_id"] = run_id
            bundle = {
                "run": {
                    "id": run_id, "season": catalog["season"], "gameweek": catalog["gameweek"],
                    "generated_at": now.isoformat(), "completed_at": datetime.now(timezone.utc).isoformat(),
                    "model_version": MODEL_VERSION,
                    "training_window": "2023-24, 2024-25, 2025-26 + 1.75x 2026-27 season-to-date",
                    "player_count": len({row["player_code"] for row in predictions}),
                },
                "roster_snapshot": {
                    "id": snapshot_id, "sha256": hashlib.sha256(bootstrap_bytes).hexdigest(),
                },
                "teams": catalog["teams"], "players": catalog["players"], "roster": catalog["roster"],
                "fixtures": catalog["fixtures"], "observations": observations, "predictions": predictions,
                "sources": [
                    {
                        "url": row.url, "retrieved_at": row.retrieved_at, "sha256": row.sha256,
                        "schema_name": row.schema_name, "local_path": row.local_path,
                    }
                    for row in client.records
                ],
            }
            if progress:
                progress("Validating and publishing the local prediction run")
            database.promote(bundle, audit_id)
            return RefreshResult(
                True, f"Gameweek {catalog['gameweek']} updated", run_id, catalog["gameweek"],
                bundle["run"]["player_count"], database.path, False,
            )
    except Exception as error:
        database.fail_audit(audit_id, str(error))
        previous = database.latest_run()
        return RefreshResult(
            False, str(error), previous["id"] if previous else None,
            previous["gameweek"] if previous else None, previous["player_count"] if previous else 0,
            database.path, previous is not None,
        )
