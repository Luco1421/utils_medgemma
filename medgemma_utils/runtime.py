"""Runtime logging helpers for scripts and SLURM jobs."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as a compact human-readable duration."""

    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


@contextmanager
def timed_stage(logger: logging.Logger, label: str) -> Iterator[dict[str, object]]:
    """Log start/end for a stage and expose timing metadata."""

    started_at = utc_now_iso()
    started = time.perf_counter()
    metadata: dict[str, object] = {
        "label": label,
        "started_at": started_at,
    }
    logger.info("%s started at %s", label, started_at)
    try:
        yield metadata
    finally:
        elapsed = time.perf_counter() - started
        finished_at = utc_now_iso()
        metadata.update({
            "finished_at": finished_at,
            "duration_seconds": round(elapsed, 3),
            "duration": format_duration(elapsed),
        })
        logger.info(
            "%s finished at %s in %s",
            label,
            finished_at,
            metadata["duration"],
        )
