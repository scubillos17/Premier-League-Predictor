from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression

from .rules import defensive_threshold, normalize_participation
from .sources import ARCHIVE_BASE, POSITION_MAP, SourceClient

MODEL_VERSION = "desktop-v1-catboost-poisson"
HISTORICAL_SEASONS = (
    ("2023-24", 1.00, False),
    ("2024-25", 1.10, False),
    ("2025-26", 1.25, False),
    ("2026-27", 1.75, True),
)
NUMERIC_COLUMNS = (
    "minutes", "starts", "goals_scored", "assists", "expected_goals", "expected_assists",
    "clean_sheets", "yellow_cards", "red_cards", "defensive_actions",
)
FEATURES = (
    "minutes_lag6", "starts_lag6", "goals_scored_lag6", "assists_lag6",
    "expected_goals_lag6", "expected_assists_lag6", "clean_sheets_lag6",
    "defensive_actions_lag6", "yellow_cards_lag10", "red_cards_lag10",
    "was_home", "opponent_strength",
)
CAT_FEATURES = ("position",)


def _progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback:
        callback(message)


def _ensure(frame: pd.DataFrame, defaults: dict[str, Any]) -> pd.DataFrame:
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value
    return frame


def _defensive_actions(frame: pd.DataFrame) -> pd.Series:
    for column in ("defensive_contribution", "defensive_actions", "clearances_blocks_interceptions"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    components = [name for name in ("tackles", "clearances", "blocks", "interceptions", "recoveries") if name in frame]
    if components:
        return sum((pd.to_numeric(frame[name], errors="coerce").fillna(0.0) for name in components), start=pd.Series(0.0, index=frame.index))
    return pd.Series(0.0, index=frame.index)


def prepare_history(frames: list[pd.DataFrame]) -> pd.DataFrame:
    history = pd.concat(frames, ignore_index=True)
    history = _ensure(
        history,
        {
            "code": 0, "element_type": 3, "minutes": 0, "starts": 0, "goals_scored": 0,
            "assists": 0, "expected_goals": 0.0, "expected_assists": 0.0, "clean_sheets": 0,
            "yellow_cards": 0, "red_cards": 0, "was_home": False, "opponent_team": 10,
            "kickoff_time": "", "season": "unknown", "season_weight": 1.0,
        },
    )
    mapped_position = pd.to_numeric(history["element_type"], errors="coerce").map(POSITION_MAP)
    existing_position = history["position"] if "position" in history else pd.Series(index=history.index, dtype=object)
    history["position"] = mapped_position.fillna(existing_position).fillna("MID").astype(str)
    history["player_code"] = pd.to_numeric(history["code"], errors="coerce").fillna(0).astype("int64").astype(str)
    history = history[history["player_code"] != "0"].copy()
    history["kickoff_time"] = pd.to_datetime(history["kickoff_time"], errors="coerce", utc=True)
    history["defensive_actions"] = _defensive_actions(history)
    for column in NUMERIC_COLUMNS:
        history[column] = pd.to_numeric(history[column], errors="coerce").fillna(0.0)
    history["was_home"] = history["was_home"].astype(bool).astype(int)
    history["opponent_strength"] = pd.to_numeric(history["opponent_team"], errors="coerce").fillna(10.0) / 20.0
    history = history.sort_values(["player_code", "kickoff_time", "season"], na_position="last").reset_index(drop=True)
    for column in NUMERIC_COLUMNS:
        window = 10 if column in {"yellow_cards", "red_cards"} else 6
        history[f"{column}_lag{window}"] = history.groupby("player_code", sort=False)[column].transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).mean()
        )
    history["appearance_class"] = np.select(
        [history["starts"] > 0, history["minutes"] > 0], [2, 1], default=0
    )
    history["season_weight"] = pd.to_numeric(history["season_weight"], errors="coerce").fillna(1.0)
    return history.dropna(subset=["minutes_lag6"]).reset_index(drop=True)


def load_history(client: SourceClient, callback: Callable[[str], None] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season, weight, optional in HISTORICAL_SEASONS:
        _progress(callback, f"Loading {season} history")
        gw = client.csv(f"{ARCHIVE_BASE}/{season}/gws/merged_gw.csv", f"{season} gameweeks", optional=optional)
        raw = client.csv(f"{ARCHIVE_BASE}/{season}/players_raw.csv", f"{season} players", optional=optional)
        if gw.empty or raw.empty:
            continue
        element_col = "element" if "element" in gw.columns else "id"
        raw_id = "id" if "id" in raw.columns else "element"
        columns = [name for name in (raw_id, "code", "element_type", "team", "first_name", "second_name", "web_name") if name in raw]
        lookup = raw[columns].drop_duplicates(raw_id)
        merged = gw.merge(lookup, left_on=element_col, right_on=raw_id, how="left", suffixes=("", "_raw"))
        teams = client.csv(f"{ARCHIVE_BASE}/{season}/teams.csv", f"{season} teams", optional=True)
        team_names: dict[int, str] = {}
        if not teams.empty and {"id", "name"}.issubset(teams.columns):
            team_names = dict(zip(pd.to_numeric(teams["id"], errors="coerce").fillna(0).astype(int), teams["name"].astype(str)))
        team_col = "team_raw" if "team_raw" in merged else "team"
        merged["club_name"] = pd.to_numeric(merged.get(team_col, 0), errors="coerce").map(team_names).fillna("Unknown")
        merged["opponent_name"] = pd.to_numeric(merged.get("opponent_team", 0), errors="coerce").map(team_names).fillna("Unknown")
        merged["season"] = season
        merged["season_weight"] = weight
        frames.append(merged)
    if len(frames) < 3:
        raise RuntimeError("Fewer than three required historical seasons were available")
    history = prepare_history(frames)
    if len(history) < 10_000:
        raise RuntimeError(f"Historical contract check failed: only {len(history)} observations")
    return history


class ConstantBinaryModel:
    def __init__(self, probability: float) -> None:
        self.probability = min(max(float(probability), 0.0001), 0.9999)

    def predict_proba(self, values: Any) -> np.ndarray:
        count = len(values)
        return np.tile([1.0 - self.probability, self.probability], (count, 1))


def _binary_model(x: pd.DataFrame, labels: pd.Series, weights: pd.Series, common: dict[str, Any]) -> Any:
    if labels.nunique() < 2:
        return ConstantBinaryModel(float(labels.mean()))
    return CatBoostClassifier(loss_function="Logloss", **common).fit(
        x, labels, sample_weight=weights, cat_features=list(CAT_FEATURES)
    )


def train_models(history: pd.DataFrame, callback: Callable[[str], None] | None = None) -> dict[str, Any]:
    _progress(callback, "Training appearance and event models")
    x = history[list(FEATURES + CAT_FEATURES)].copy()
    x["position"] = x["position"].fillna("MID").astype(str)
    x[list(FEATURES)] = x[list(FEATURES)].fillna(0.0)
    weights = history["season_weight"]
    common = {
        "iterations": 150, "depth": 6, "learning_rate": 0.055, "verbose": False,
        "random_seed": 42, "allow_writing_files": False, "thread_count": -1,
    }
    models: dict[str, Any] = {}
    models["appearance"] = CatBoostClassifier(loss_function="MultiClass", **common).fit(
        x, history["appearance_class"], sample_weight=weights, cat_features=list(CAT_FEATURES)
    )
    models["minutes"] = CatBoostRegressor(loss_function="RMSE", **common).fit(
        x, history["minutes"], sample_weight=weights, cat_features=list(CAT_FEATURES)
    )
    for target in ("goals_scored", "assists", "defensive_actions"):
        models[target] = CatBoostRegressor(loss_function="Poisson", **common).fit(
            x, history[target], sample_weight=weights, cat_features=list(CAT_FEATURES)
        )
    models["clean_sheets"] = _binary_model(x, (history["clean_sheets"] > 0).astype(int), weights, common)
    models["yellow_cards"] = _binary_model(x, (history["yellow_cards"] > 0).astype(int), weights, common)
    red_labels = (history["red_cards"] > 0).astype(int)
    red_x = pd.get_dummies(x, columns=list(CAT_FEATURES), dtype=float)
    models["red_columns"] = list(red_x.columns)
    if red_labels.nunique() < 2:
        models["red_cards"] = ConstantBinaryModel(float(red_labels.mean()))
    else:
        models["red_cards"] = LogisticRegression(C=0.15, class_weight="balanced", max_iter=1000).fit(
            red_x, red_labels, sample_weight=weights
        )
    return models


def build_current_features(catalog: dict[str, Any], history: pd.DataFrame) -> pd.DataFrame:
    latest = history.groupby("player_code", sort=False).tail(1).set_index("player_code")
    rows: list[dict[str, Any]] = []
    for player in catalog["candidates"]:
        base = latest.loc[player["code"]].to_dict() if player["code"] in latest.index else {}
        starts = max(player["starts_total"], 1.0)
        fallback = {
            "minutes_lag6": min(player["minutes_total"] / starts, 90.0),
            "starts_lag6": min(player["starts_total"] / max(catalog["gameweek"] - 1, 1), 1.0),
            "goals_scored_lag6": player["goals_total"] / starts,
            "assists_lag6": player["assists_total"] / starts,
            "expected_goals_lag6": player["xg_total"] / starts,
            "expected_assists_lag6": player["xa_total"] / starts,
            "clean_sheets_lag6": player["clean_total"] / starts,
            "defensive_actions_lag6": 0.0,
            "yellow_cards_lag10": player["yellow_total"] / starts,
            "red_cards_lag10": player["red_total"] / starts,
        }
        for fixture in catalog["fixtures"]:
            if player["team_id"] not in {fixture["home_team_id"], fixture["away_team_id"]}:
                continue
            home = fixture["home_team_id"] == player["team_id"]
            opponent_id = fixture["away_team_id"] if home else fixture["home_team_id"]
            opponent = next(team for team in catalog["teams"] if team["fpl_id"] == opponent_id)
            row = {feature: float(base.get(feature, fallback.get(feature, 0.0)) or 0.0) for feature in FEATURES}
            row.update(
                {
                    "player_code": player["code"], "player_name": player["name"], "position": player["position"],
                    "team_code": player["team_code"], "team_name": player["team_name"], "status": player["status"],
                    "chance_next": player["chance_next"], "fixture_code": fixture["code"],
                    "opponent_name": opponent["name"], "opponent_code": opponent["code"],
                    "venue": "H" if home else "A", "was_home": int(home),
                    "opponent_strength": float(3.0 / 5.0), "kickoff_at": fixture["kickoff_at"],
                }
            )
            rows.append(row)
    if not rows:
        raise RuntimeError("The current roster produced no prediction candidates")
    return pd.DataFrame(rows)


def _poisson_tail(rate: float, threshold: int) -> float:
    rate = max(float(rate), 0.0)
    below = sum(math.exp(-rate) * rate**k / math.factorial(k) for k in range(threshold))
    return min(max(1.0 - below, 0.0), 1.0)


def score(models: dict[str, Any], current: pd.DataFrame) -> list[dict[str, Any]]:
    x = current[list(FEATURES + CAT_FEATURES)].copy()
    x[list(FEATURES)] = x[list(FEATURES)].fillna(0.0)
    x["position"] = x["position"].fillna("MID").astype(str)
    appearance_raw = models["appearance"].predict_proba(x)
    appearance_classes = [int(value) for value in models["appearance"].classes_]
    class_index = {value: index for index, value in enumerate(appearance_classes)}
    minutes = np.clip(models["minutes"].predict(x), 0, 90)
    goals = np.clip(models["goals_scored"].predict(x), 0, 2.5)
    assists = np.clip(models["assists"].predict(x), 0, 2.0)
    defensive = np.clip(models["defensive_actions"].predict(x), 0, 35.0)
    clean = models["clean_sheets"].predict_proba(x)[:, 1]
    yellow = models["yellow_cards"].predict_proba(x)[:, 1]
    red_x = pd.get_dummies(x, columns=list(CAT_FEATURES), dtype=float).reindex(columns=models["red_columns"], fill_value=0.0)
    red = np.clip(models["red_cards"].predict_proba(red_x)[:, 1] * 0.12, 0.0001, 0.08)
    output: list[dict[str, Any]] = []
    for index, row in current.reset_index(drop=True).iterrows():
        start_raw = float(appearance_raw[index][class_index.get(2, -1)]) if 2 in class_index else 0.0
        bench_raw = float(appearance_raw[index][class_index.get(1, -1)]) if 1 in class_index else 0.0
        availability = 1.0
        if pd.notna(row["chance_next"]):
            availability = min(max(float(row["chance_next"]) / 100.0, 0.0), 1.0)
        if row["status"] == "u":
            availability *= 0.05
        start, bench, no_show = normalize_participation(start_raw * availability, bench_raw * availability)
        threshold = defensive_threshold(str(row["position"]))
        defensive_probability = 0.0 if threshold is None else _poisson_tail(defensive[index], threshold) * start
        confidence = "high" if row["minutes_lag6"] >= 70 else "medium" if row["minutes_lag6"] >= 35 else "low"
        output.append(
            {
                "player_code": row["player_code"], "fixture_code": row["fixture_code"],
                "expected_goals": round(float(goals[index]), 5),
                "goals_if_start": round(float(goals[index] / max(start, 0.25)), 5),
                "expected_assists": round(float(assists[index]), 5),
                "assists_if_start": round(float(assists[index] / max(start, 0.25)), 5),
                "defensive_actions": round(float(defensive[index]), 4),
                "defensive_return_probability": round(float(defensive_probability), 6),
                "clean_sheet_probability": round(float(clean[index] * (start + bench * 0.2)), 6),
                "yellow_probability": round(float(yellow[index] * (start + bench)), 6),
                "red_probability": round(float(red[index] * (start + bench)), 6),
                "start_probability": start, "bench_probability": bench,
                "no_appearance_probability": no_show,
                "expected_minutes": round(float(minutes[index] * availability), 2),
                "confidence": confidence,
                "explanations_json": json.dumps(
                    ["Strictly lagged six-match form", "Opponent and home/away context", "Availability-adjusted expected minutes"]
                ),
            }
        )
    return output


def observation_rows(history: pd.DataFrame, active_codes: set[str]) -> list[dict[str, Any]]:
    selected = history[history["player_code"].isin(active_codes)].copy()
    rows: list[dict[str, Any]] = []
    for index, row in selected.iterrows():
        kickoff = row["kickoff_time"]
        rows.append(
            {
                "player_code": row["player_code"], "season": str(row["season"]),
                "gameweek": int(row.get("GW", row.get("round", 0)) or 0),
                "fixture_code": f"{row['season']}-{row.get('fixture', index)}",
                "kickoff_at": kickoff.isoformat() if pd.notna(kickoff) else None,
                "club_name": str(row.get("club_name", "Unknown")),
                "opponent_name": str(row.get("opponent_name", "Unknown")),
                "venue": "H" if bool(row["was_home"]) else "A", "minutes": float(row["minutes"]),
                "starts": float(row["starts"]), "goals": float(row["goals_scored"]),
                "assists": float(row["assists"]), "expected_goals": float(row["expected_goals"]),
                "expected_assists": float(row["expected_assists"]),
                "defensive_actions": float(row["defensive_actions"]),
                "clean_sheet": int(row["clean_sheets"] > 0), "yellow_card": int(row["yellow_cards"] > 0),
                "red_card": int(row["red_cards"] > 0),
            }
        )
    return rows


def create_forecasts(
    catalog: dict[str, Any], client: SourceClient, callback: Callable[[str], None] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    history = load_history(client, callback)
    models = train_models(history, callback)
    _progress(callback, "Scoring the upcoming gameweek")
    current = build_current_features(catalog, history)
    predictions = score(models, current)
    observations = observation_rows(history, {row["code"] for row in catalog["players"]})
    return predictions, observations
