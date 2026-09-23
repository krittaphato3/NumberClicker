"""Run statistics for the number-sequence clicking bot."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RunStats:
    """Counters + derived metrics for one bot run (thread-safe enough for GUI polling)."""

    started_at: float = field(default_factory=time.time)
    clicks: int = 0
    hits: int = 0
    misses: int = 0
    retries: int = 0

    def reset(self) -> None:
        """Zero counters and restart the clock."""
        self.started_at = time.time()
        self.clicks = 0
        self.hits = 0
        self.misses = 0
        self.retries = 0

    def record_hit(self) -> None:
        """Record one successful click."""
        self.clicks += 1
        self.hits += 1

    def record_miss(self, retried: bool = False) -> None:
        """Record one failed click; optionally count a retry."""
        self.clicks += 1
        self.misses += 1
        if retried:
            self.retries += 1

    def elapsed(self) -> float:
        """Seconds since start (never negative)."""
        return max(0.0, time.time() - self.started_at)

    def cps(self) -> float:
        """Clicks per second."""
        e = self.elapsed()
        return (self.clicks / e) if e > 0 else 0.0

    def accuracy(self) -> float:
        """hits / clicks in [0, 1]; 0.0 when no clicks yet."""
        if self.clicks <= 0:
            return 0.0
        return max(0.0, min(1.0, self.hits / self.clicks))

    def to_dict(self) -> dict:
        """Plain-dict snapshot for queues / JSON."""
        return {
            "clicks": self.clicks,
            "hits": self.hits,
            "misses": self.misses,
            "retries": self.retries,
            "elapsed": self.elapsed(),
            "cps": self.cps(),
            "accuracy": self.accuracy(),
        }

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"clicks={self.clicks} hits={self.hits} misses={self.misses} "
            f"retries={self.retries} elapsed={self.elapsed():.1f}s "
            f"cps={self.cps():.2f} acc={self.accuracy() * 100:.1f}%"
        )
