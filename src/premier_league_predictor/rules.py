from __future__ import annotations

import math


def defensive_threshold(position: str) -> int | None:
    """Return the FPL defensive-contribution threshold for a position."""
    return {"DEF": 10, "MID": 12, "FWD": 12}.get(position)


def normalize_participation(start: float, bench: float) -> tuple[float, float, float]:
    start = min(max(float(start), 0.0), 1.0)
    bench = min(max(float(bench), 0.0), 1.0 - start)
    no_show = max(0.0, 1.0 - start - bench)
    total = start + bench + no_show
    return tuple(round(value / total, 6) for value in (start, bench, no_show))


def poisson_at_least_one(rate: float) -> float:
    return 1.0 - math.exp(-max(float(rate), 0.0))


def validate_probability(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 1.0


class CostGuard:
    forbidden_environment_variables = {
        "OPENAI_API_KEY",
        "FOOTBALL_DATA_API_KEY",
        "SPORTMONKS_API_KEY",
        "API_FOOTBALL_KEY",
    }

    def assert_zero_cost(self, environment: dict[str, str] | None = None) -> None:
        import os

        source = os.environ if environment is None else environment
        enabled = sorted(name for name in self.forbidden_environment_variables if source.get(name))
        if enabled:
            raise RuntimeError(f"Paid or runtime API credentials are not permitted: {', '.join(enabled)}")
