from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1


def application_data_dir() -> Path:
    override = os.environ.get("PLP_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return root / "PremierLeaguePredictor"


def default_database_path() -> Path:
    override = os.environ.get("PLP_DATABASE_PATH")
    return Path(override).expanduser().resolve() if override else application_data_dir() / "predictor.db"


class Database:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else default_database_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
        finally:
            connection.close()

    def migrate(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS teams (
                    code TEXT PRIMARY KEY,
                    fpl_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    short_name TEXT NOT NULL,
                    season TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS players (
                    code TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    position TEXT NOT NULL CHECK(position IN ('GK','DEF','MID','FWD')),
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS roster_snapshots (
                    id TEXT PRIMARY KEY,
                    season TEXT NOT NULL,
                    gameweek INTEGER NOT NULL,
                    captured_at TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS current_roster (
                    snapshot_id TEXT NOT NULL REFERENCES roster_snapshots(id) ON DELETE CASCADE,
                    player_code TEXT NOT NULL REFERENCES players(code),
                    team_code TEXT NOT NULL REFERENCES teams(code),
                    status TEXT NOT NULL,
                    PRIMARY KEY(snapshot_id, player_code)
                );
                CREATE TABLE IF NOT EXISTS fixtures (
                    code TEXT PRIMARY KEY,
                    season TEXT NOT NULL,
                    gameweek INTEGER NOT NULL,
                    kickoff_at TEXT,
                    home_team_code TEXT NOT NULL REFERENCES teams(code),
                    away_team_code TEXT NOT NULL REFERENCES teams(code),
                    started INTEGER NOT NULL DEFAULT 0,
                    finished INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS observations (
                    player_code TEXT NOT NULL REFERENCES players(code),
                    season TEXT NOT NULL,
                    gameweek INTEGER NOT NULL,
                    fixture_code TEXT NOT NULL,
                    kickoff_at TEXT,
                    club_name TEXT NOT NULL,
                    opponent_name TEXT NOT NULL,
                    venue TEXT NOT NULL,
                    minutes REAL NOT NULL,
                    starts REAL NOT NULL,
                    goals REAL NOT NULL,
                    assists REAL NOT NULL,
                    expected_goals REAL NOT NULL,
                    expected_assists REAL NOT NULL,
                    defensive_actions REAL NOT NULL,
                    clean_sheet INTEGER NOT NULL,
                    yellow_card INTEGER NOT NULL,
                    red_card INTEGER NOT NULL,
                    PRIMARY KEY(player_code, season, fixture_code)
                );
                CREATE TABLE IF NOT EXISTS prediction_runs (
                    id TEXT PRIMARY KEY,
                    season TEXT NOT NULL,
                    gameweek INTEGER NOT NULL,
                    generated_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    roster_snapshot_id TEXT NOT NULL REFERENCES roster_snapshots(id),
                    model_version TEXT NOT NULL,
                    training_window TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('complete','failed')),
                    player_count INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    run_id TEXT NOT NULL REFERENCES prediction_runs(id) ON DELETE CASCADE,
                    player_code TEXT NOT NULL REFERENCES players(code),
                    fixture_code TEXT NOT NULL REFERENCES fixtures(code),
                    expected_goals REAL NOT NULL,
                    goals_if_start REAL NOT NULL,
                    expected_assists REAL NOT NULL,
                    assists_if_start REAL NOT NULL,
                    defensive_actions REAL NOT NULL,
                    defensive_return_probability REAL NOT NULL,
                    clean_sheet_probability REAL NOT NULL,
                    yellow_probability REAL NOT NULL,
                    red_probability REAL NOT NULL,
                    start_probability REAL NOT NULL,
                    bench_probability REAL NOT NULL,
                    no_appearance_probability REAL NOT NULL,
                    expected_minutes REAL NOT NULL,
                    confidence TEXT NOT NULL,
                    explanations_json TEXT NOT NULL,
                    PRIMARY KEY(run_id, player_code, fixture_code)
                );
                CREATE TABLE IF NOT EXISTS refresh_audits (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    run_id TEXT,
                    message TEXT NOT NULL,
                    source_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS source_cache (
                    url TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    schema_name TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    PRIMARY KEY(url, sha256)
                );
                CREATE INDEX IF NOT EXISTS idx_runs_latest ON prediction_runs(status, generated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_predictions_rank_xg ON predictions(run_id, expected_goals DESC);
                CREATE INDEX IF NOT EXISTS idx_predictions_rank_xa ON predictions(run_id, expected_assists DESC);
                CREATE INDEX IF NOT EXISTS idx_predictions_player ON predictions(player_code, run_id);
                CREATE INDEX IF NOT EXISTS idx_observations_opponent ON observations(player_code, opponent_name, kickoff_at DESC);
                CREATE INDEX IF NOT EXISTS idx_fixtures_gameweek ON fixtures(season, gameweek, kickoff_at);
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()),
            )
            connection.commit()

    def begin_audit(self, audit_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO refresh_audits(id, started_at, status, message) VALUES (?, ?, 'running', '')",
                (audit_id, datetime.now(timezone.utc).isoformat()),
            )
            connection.commit()

    def fail_audit(self, audit_id: str, message: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE refresh_audits SET finished_at=?, status='failed', message=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), message[:1000], audit_id),
            )
            connection.commit()

    def promote(self, bundle: dict[str, Any], audit_id: str) -> None:
        run = bundle["run"]
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for team in bundle["teams"]:
                connection.execute(
                    """INSERT INTO teams(code,fpl_id,name,short_name,season,active) VALUES(?,?,?,?,?,1)
                    ON CONFLICT(code) DO UPDATE SET fpl_id=excluded.fpl_id,name=excluded.name,
                    short_name=excluded.short_name,season=excluded.season,active=1""",
                    (team["code"], team["fpl_id"], team["name"], team["short_name"], run["season"]),
                )
            connection.execute("UPDATE teams SET active=0 WHERE season<>?", (run["season"],))
            for player in bundle["players"]:
                connection.execute(
                    """INSERT INTO players(code,name,position,first_seen,last_seen) VALUES(?,?,?,?,?)
                    ON CONFLICT(code) DO UPDATE SET name=excluded.name,position=excluded.position,last_seen=excluded.last_seen""",
                    (player["code"], player["name"], player["position"], run["generated_at"], run["generated_at"]),
                )
            snapshot = bundle["roster_snapshot"]
            connection.execute(
                "INSERT INTO roster_snapshots(id,season,gameweek,captured_at,source_sha256) VALUES(?,?,?,?,?)",
                (snapshot["id"], run["season"], run["gameweek"], run["generated_at"], snapshot["sha256"]),
            )
            connection.executemany(
                "INSERT INTO current_roster(snapshot_id,player_code,team_code,status) VALUES(?,?,?,?)",
                [(snapshot["id"], row["player_code"], row["team_code"], row["status"]) for row in bundle["roster"]],
            )
            for fixture in bundle["fixtures"]:
                connection.execute(
                    """INSERT INTO fixtures(code,season,gameweek,kickoff_at,home_team_code,away_team_code,started,finished)
                    VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(code) DO UPDATE SET season=excluded.season,
                    gameweek=excluded.gameweek,kickoff_at=excluded.kickoff_at,home_team_code=excluded.home_team_code,
                    away_team_code=excluded.away_team_code,started=excluded.started,finished=excluded.finished""",
                    (
                        fixture["code"], run["season"], fixture["gameweek"], fixture.get("kickoff_at"),
                        fixture["home_team_code"], fixture["away_team_code"], int(fixture["started"]), int(fixture["finished"]),
                    ),
                )
            for observation in bundle.get("observations", []):
                connection.execute(
                    """INSERT OR REPLACE INTO observations(player_code,season,gameweek,fixture_code,kickoff_at,
                    club_name,opponent_name,venue,minutes,starts,goals,assists,expected_goals,expected_assists,
                    defensive_actions,clean_sheet,yellow_card,red_card) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    tuple(observation[key] for key in (
                        "player_code", "season", "gameweek", "fixture_code", "kickoff_at", "club_name",
                        "opponent_name", "venue", "minutes", "starts", "goals", "assists", "expected_goals",
                        "expected_assists", "defensive_actions", "clean_sheet", "yellow_card", "red_card",
                    )),
                )
            connection.execute(
                """INSERT INTO prediction_runs(id,season,gameweek,generated_at,completed_at,roster_snapshot_id,
                model_version,training_window,status,player_count) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    run["id"], run["season"], run["gameweek"], run["generated_at"], run["completed_at"],
                    snapshot["id"], run["model_version"], run["training_window"], "complete", run["player_count"],
                ),
            )
            connection.executemany(
                """INSERT INTO predictions(run_id,player_code,fixture_code,expected_goals,goals_if_start,
                expected_assists,assists_if_start,defensive_actions,defensive_return_probability,
                clean_sheet_probability,yellow_probability,red_probability,start_probability,bench_probability,
                no_appearance_probability,expected_minutes,confidence,explanations_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    tuple(row[key] for key in (
                        "run_id", "player_code", "fixture_code", "expected_goals", "goals_if_start",
                        "expected_assists", "assists_if_start", "defensive_actions", "defensive_return_probability",
                        "clean_sheet_probability", "yellow_probability", "red_probability", "start_probability",
                        "bench_probability", "no_appearance_probability", "expected_minutes", "confidence",
                        "explanations_json",
                    ))
                    for row in bundle["predictions"]
                ],
            )
            connection.executemany(
                "INSERT OR IGNORE INTO source_cache(url,retrieved_at,sha256,schema_name,local_path) VALUES(?,?,?,?,?)",
                [
                    (row["url"], row["retrieved_at"], row["sha256"], row["schema_name"], row["local_path"])
                    for row in bundle.get("sources", [])
                ],
            )
            connection.execute(
                "UPDATE refresh_audits SET finished_at=?,status='complete',run_id=?,message=?,source_count=? WHERE id=?",
                (run["completed_at"], run["id"], f"Published {run['player_count']} players", len(bundle.get("sources", [])), audit_id),
            )
            connection.commit()

    def latest_run(self) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM prediction_runs WHERE status='complete' ORDER BY generated_at DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def latest_error(self) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT message FROM refresh_audits WHERE status='failed' ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            return row["message"] if row else None

    def dashboard(self) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
        run = self.latest_run()
        if not run:
            return None, [], []
        with self.connect() as connection:
            fixture_rows = [
                dict(row) for row in connection.execute(
                    """SELECT pr.*,p.name,p.position,t.name AS club,
                    CASE WHEN f.home_team_code=cr.team_code THEN away.name ELSE home.name END AS opponent,
                    CASE WHEN f.home_team_code=cr.team_code THEN 'H' ELSE 'A' END AS venue,f.kickoff_at
                    FROM predictions pr
                    JOIN prediction_runs r ON r.id=pr.run_id
                    JOIN current_roster cr ON cr.snapshot_id=r.roster_snapshot_id AND cr.player_code=pr.player_code
                    JOIN players p ON p.code=pr.player_code
                    JOIN teams t ON t.code=cr.team_code
                    JOIN fixtures f ON f.code=pr.fixture_code
                    JOIN teams home ON home.code=f.home_team_code
                    JOIN teams away ON away.code=f.away_team_code
                    WHERE pr.run_id=? ORDER BY p.name,f.kickoff_at""",
                    (run["id"],),
                )
            ]
            fixtures = [
                dict(row) for row in connection.execute(
                    """SELECT f.*,home.name AS home_name,home.short_name AS home_short,
                    away.name AS away_name,away.short_name AS away_short FROM fixtures f
                    JOIN teams home ON home.code=f.home_team_code JOIN teams away ON away.code=f.away_team_code
                    WHERE f.season=? AND f.gameweek=? AND f.started=0 ORDER BY f.kickoff_at""",
                    (run["season"], run["gameweek"]),
                )
            ]
        grouped: dict[str, dict[str, Any]] = {}
        for row in fixture_rows:
            item = grouped.setdefault(
                row["player_code"],
                {
                    "player_code": row["player_code"], "name": row["name"], "position": row["position"],
                    "club": row["club"], "opponents": [], "venues": [], "expected_goals": 0.0,
                    "goals_if_start": 0.0, "expected_assists": 0.0, "assists_if_start": 0.0,
                    "defensive_actions": 0.0, "defensive_return_probability": 0.0,
                    "clean_sheet_probability": 0.0, "yellow_probability": 0.0, "red_probability": 0.0,
                    "start_probability": 0.0, "bench_probability": 0.0, "no_appearance_probability": 0.0,
                    "expected_minutes": 0.0, "confidence": row["confidence"], "drivers": [], "fixtures": [],
                },
            )
            item["opponents"].append(row["opponent"])
            item["venues"].append(row["venue"])
            for key in ("expected_goals", "goals_if_start", "expected_assists", "assists_if_start", "defensive_actions", "expected_minutes"):
                item[key] += float(row[key])
            for key in ("defensive_return_probability", "clean_sheet_probability", "yellow_probability", "red_probability"):
                item[key] = 1.0 - (1.0 - item[key]) * (1.0 - float(row[key]))
            for key in ("start_probability", "bench_probability", "no_appearance_probability"):
                item[key] = max(item[key], float(row[key]))
            item["drivers"] = json.loads(row["explanations_json"])
            item["fixtures"].append(dict(row))
        return run, list(grouped.values()), fixtures

    def player_history(self, player_code: str, opponent: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with self.connect() as connection:
            recent = [dict(row) for row in connection.execute(
                "SELECT * FROM observations WHERE player_code=? ORDER BY kickoff_at DESC LIMIT 10", (player_code,)
            )]
            versus: list[dict[str, Any]] = []
            if opponent:
                versus = [dict(row) for row in connection.execute(
                    """SELECT * FROM observations WHERE player_code=? AND lower(opponent_name)=lower(?)
                    ORDER BY kickoff_at DESC LIMIT 20""", (player_code, opponent)
                )]
            return recent, versus
