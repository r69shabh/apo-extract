"""Token/cost/latency accounting and multi-dimensional budget enforcement."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..config import BudgetConfig


class BudgetExceeded(Exception):
    pass


@dataclass
class CallRecord:
    role: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int


class BudgetTracker:
    """Tracks spend across a run; checked before every iteration and LLM batch."""

    def __init__(self, config: BudgetConfig, *, spent_tokens: int = 0, spent_usd: float = 0.0, started_at: float | None = None):
        self.config = config
        self.spent_tokens = spent_tokens
        self.spent_usd = spent_usd
        self.started_at = started_at if started_at is not None else time.time()

    @property
    def spent_seconds(self) -> float:
        return time.time() - self.started_at

    def record(self, call: CallRecord) -> None:
        self.spent_tokens += call.input_tokens + call.output_tokens
        self.spent_usd += call.cost_usd

    def remaining(self) -> dict:
        return {
            "tokens": self.config.max_total_tokens - self.spent_tokens,
            "usd": self.config.max_cost_usd - self.spent_usd,
            "seconds": self.config.max_wall_clock_seconds - self.spent_seconds,
        }

    def exceeded(self) -> str | None:
        if self.spent_tokens >= self.config.max_total_tokens:
            return "max_total_tokens"
        if self.spent_usd >= self.config.max_cost_usd:
            return "max_cost_usd"
        if self.spent_seconds >= self.config.max_wall_clock_seconds:
            return "max_wall_clock_seconds"
        return None

    def check(self) -> None:
        reason = self.exceeded()
        if reason:
            raise BudgetExceeded(reason)
