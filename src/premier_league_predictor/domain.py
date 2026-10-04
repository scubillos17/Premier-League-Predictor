from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

Position = Literal["GK", "DEF", "MID", "FWD"]
Confidence = Literal["low", "medium", "high"]


@dataclass(frozen=True, slots=True)
class FixtureForecast:
    fixture_code: str
    opponent: str
    venue: Literal["H", "A"]
    kickoff_at: str
    expected_goals: float
    expected_assists: float
    defensive_actions: float
    defensive_return_probability: float


@dataclass(frozen=True, slots=True)
class PlayerForecast:
    player_code: str
    name: str
    club: str
    position: Position
    opponent: str
    venue: str
    expected_goals: float
    goals_if_start: float
    expected_assists: float
    assists_if_start: float
    defensive_actions: float
    defensive_return_probability: float
    clean_sheet_probability: float
    yellow_probability: float
    red_probability: float
    start_probability: float
    bench_probability: float
    no_appearance_probability: float
    expected_minutes: float
    confidence: Confidence
    drivers: tuple[str, ...] = ()
    fixtures: tuple[FixtureForecast, ...] = ()


@dataclass(frozen=True, slots=True)
class PredictionRun:
    run_id: str
    season: str
    gameweek: int
    generated_at: datetime
    model_version: str
    player_count: int


@dataclass(frozen=True, slots=True)
class RefreshResult:
    ok: bool
    message: str
    run_id: str | None = None
    gameweek: int | None = None
    player_count: int = 0
    database_path: Path | None = None
    stale: bool = False
