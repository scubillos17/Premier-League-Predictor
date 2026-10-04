from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from .database import application_data_dir

FPL_BASE = "https://fantasy.premierleague.com/api"
ARCHIVE_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


@dataclass(frozen=True, slots=True)
class SourceRecord:
    url: str
    retrieved_at: str
    sha256: str
    schema_name: str
    local_path: str


class SourceClient:
    def __init__(self, cache_dir: Path | None = None) -> None:
        override = os.environ.get("PLP_CACHE_DIR")
        self.cache_dir = Path(override) if override else (cache_dir or application_data_dir() / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[SourceRecord] = []

    def _cache_path(self, url: str, suffix: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}{suffix}"

    def fetch(self, url: str, schema_name: str, *, allow_cached_on_error: bool = True) -> bytes:
        suffix = ".json" if url.endswith("/") or ".json" in url else ".csv"
        path = self._cache_path(url, suffix)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "pl-predictor-desktop/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                content = response.read()
            if not content:
                raise RuntimeError(f"Empty response from {url}")
            path.write_bytes(content)
        except (OSError, urllib.error.URLError) as error:
            if not allow_cached_on_error or not path.exists():
                raise RuntimeError(f"Unable to retrieve {schema_name}: {error}") from error
            content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        self.records.append(
            SourceRecord(url, datetime.now(timezone.utc).isoformat(), digest, schema_name, str(path))
        )
        return content

    def json(self, url: str, schema_name: str, *, allow_cached_on_error: bool = True) -> Any:
        try:
            return json.loads(self.fetch(url, schema_name, allow_cached_on_error=allow_cached_on_error))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid JSON from {schema_name}") from error

    def csv(self, url: str, schema_name: str, *, optional: bool = False) -> pd.DataFrame:
        try:
            content = self.fetch(url, schema_name, allow_cached_on_error=True)
            return pd.read_csv(BytesIO(content))
        except Exception:
            if optional:
                return pd.DataFrame()
            raise


def season_label(now: datetime | None = None) -> str:
    value = now or datetime.now(timezone.utc)
    first = value.year if value.month >= 7 else value.year - 1
    return f"{first}-{str(first + 1)[-2:]}"


def validate_current_feed(bootstrap: dict[str, Any], fixtures: list[dict[str, Any]]) -> None:
    if not isinstance(bootstrap.get("events"), list) or not bootstrap["events"]:
        raise RuntimeError("Current FPL feed has no events")
    if not isinstance(bootstrap.get("teams"), list) or len(bootstrap["teams"]) != 20:
        raise RuntimeError("Current FPL feed must contain exactly 20 teams")
    if not isinstance(bootstrap.get("elements"), list) or len(bootstrap["elements"]) < 300:
        raise RuntimeError("Current FPL feed has an incomplete player roster")
    if not isinstance(fixtures, list) or not fixtures:
        raise RuntimeError("Current FPL fixture feed is empty")
    required_player = {"code", "web_name", "team", "element_type", "status"}
    if any(not required_player.issubset(element) for element in bootstrap["elements"]):
        raise RuntimeError("Current FPL player schema is incomplete")


def next_gameweek(bootstrap: dict[str, Any], fixtures: list[dict[str, Any]]) -> int:
    declared = next((int(event["id"]) for event in bootstrap["events"] if event.get("is_next")), None)
    if declared is not None:
        return declared
    upcoming = sorted({int(row["event"]) for row in fixtures if row.get("event") and not row.get("started")})
    if not upcoming:
        raise RuntimeError("No upcoming Premier League gameweek is available")
    return upcoming[0]


def build_current_catalog(
    bootstrap: dict[str, Any], fixtures_feed: list[dict[str, Any]], *, season: str | None = None
) -> dict[str, Any]:
    validate_current_feed(bootstrap, fixtures_feed)
    current_season = season or season_label()
    gameweek = next_gameweek(bootstrap, fixtures_feed)
    teams_by_id = {int(row["id"]): row for row in bootstrap["teams"]}
    team_code_by_id = {team_id: str(row["code"]) for team_id, row in teams_by_id.items()}
    upcoming = [row for row in fixtures_feed if row.get("event") == gameweek and not row.get("started")]
    participating_team_ids = {int(row[key]) for row in upcoming for key in ("team_h", "team_a")}
    if not upcoming:
        raise RuntimeError("Upcoming gameweek has no unstarted fixtures")

    teams = [
        {
            "code": str(row["code"]),
            "fpl_id": int(row["id"]),
            "name": str(row["name"]),
            "short_name": str(row["short_name"]),
        }
        for row in bootstrap["teams"]
    ]
    players: list[dict[str, Any]] = []
    roster: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for element in bootstrap["elements"]:
        team_id = int(element["team"])
        # FPL keeps transferred/loaned/de-registered players in elements with status "u".
        # They remain useful historically but are not current prediction candidates.
        if (
            team_id not in teams_by_id
            or team_id not in participating_team_ids
            or str(element.get("status", "a")) == "u"
        ):
            continue
        player = {
            "code": str(element["code"]),
            "name": str(element["web_name"]),
            "position": POSITION_MAP.get(int(element["element_type"]), "MID"),
        }
        players.append(player)
        roster.append(
            {
                "player_code": player["code"],
                "team_code": team_code_by_id[team_id],
                "status": str(element.get("status", "a")),
            }
        )
        candidates.append(
            {
                **player,
                "team_id": team_id,
                "team_code": team_code_by_id[team_id],
                "team_name": str(teams_by_id[team_id]["name"]),
                "status": str(element.get("status", "a")),
                "chance_next": element.get("chance_of_playing_next_round"),
                "minutes_total": float(element.get("minutes", 0) or 0),
                "starts_total": float(element.get("starts", 0) or 0),
                "goals_total": float(element.get("goals_scored", 0) or 0),
                "assists_total": float(element.get("assists", 0) or 0),
                "xg_total": float(element.get("expected_goals", 0) or 0),
                "xa_total": float(element.get("expected_assists", 0) or 0),
                "clean_total": float(element.get("clean_sheets", 0) or 0),
                "yellow_total": float(element.get("yellow_cards", 0) or 0),
                "red_total": float(element.get("red_cards", 0) or 0),
            }
        )

    fixtures = [
        {
            "code": str(row["id"]),
            "gameweek": gameweek,
            "kickoff_at": row.get("kickoff_time"),
            "home_team_code": team_code_by_id[int(row["team_h"])],
            "away_team_code": team_code_by_id[int(row["team_a"])],
            "started": bool(row.get("started")),
            "finished": bool(row.get("finished")),
            "home_team_id": int(row["team_h"]),
            "away_team_id": int(row["team_a"]),
        }
        for row in upcoming
    ]
    if len(players) < 300:
        raise RuntimeError(f"Current-roster gate retained only {len(players)} players")
    return {
        "season": current_season,
        "gameweek": gameweek,
        "teams": teams,
        "players": players,
        "roster": roster,
        "candidates": candidates,
        "fixtures": fixtures,
    }
