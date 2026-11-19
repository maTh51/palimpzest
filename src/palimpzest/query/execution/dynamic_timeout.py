"""Per-logical-operator dynamic sample timeout for Abacus sentinel execution.

Instead of one fixed SAMPLE_TIMEOUT (30/60/120 s) for every operator, each logical
operator gets its own timeout derived from the latencies its physical operators have
actually shown so far:

    timeout(op) = clip( MULT * quantile_Q(successful latencies of op), MIN, MAX )

Until an operator has WARMUP successful samples, the fallback is used
(SAMPLE_TIMEOUT if set, otherwise MAX). Physical operators of the same logical op do the
same task with different models/strategies, so "much slower than its peers" is exactly
what we want to cut.

Latency is measured from when a sample STARTS running, not when it is queued, so queue
wait under a small worker pool does not count against an operator.

Env vars:
    TIMEOUT_MODE          fixed | dynamic | none   (default: fixed -> old behaviour)
    SAMPLE_TIMEOUT        fixed timeout / dynamic fallback (seconds)
    DYN_TIMEOUT_QUANTILE  default 0.90
    DYN_TIMEOUT_MULT      default 1.5
    DYN_TIMEOUT_MIN       default 10
    DYN_TIMEOUT_MAX       default 300
    DYN_TIMEOUT_WARMUP    default 5   (successful samples before going dynamic)
    TIMEOUT_LOG_CSV       if set, one row per sample is appended (for plotting)
"""
from __future__ import annotations

import csv
import os
import threading
import time

import numpy as np

_CSV_FIELDS = [
    "wall_time", "mode", "logical_op_id", "full_op_id", "outcome",
    "latency_s", "queue_wait_s", "timeout_s", "timeout_source", "n_obs_for_op",
]


def _env_float(name: str, default: float | None) -> float | None:
    v = os.getenv(name)
    if v in (None, ""):
        return default
    try:
        return float(v)
    except ValueError:
        return default


class DynamicTimeout:
    _instance: DynamicTimeout | None = None
    _lock = threading.Lock()

    def __init__(self):
        self.mode = os.getenv("TIMEOUT_MODE", "fixed").lower()
        self.fixed = _env_float("SAMPLE_TIMEOUT", None)
        if self.fixed is not None and self.fixed <= 0:
            self.fixed = None
        self.q = _env_float("DYN_TIMEOUT_QUANTILE", 0.90)
        self.mult = _env_float("DYN_TIMEOUT_MULT", 1.5)
        self.t_min = _env_float("DYN_TIMEOUT_MIN", 10.0)
        self.t_max = _env_float("DYN_TIMEOUT_MAX", 300.0)
        self.warmup = int(_env_float("DYN_TIMEOUT_WARMUP", 5))
        self.csv_path = os.getenv("TIMEOUT_LOG_CSV")
        self._lat: dict[str, list[float]] = {}
        self._io_lock = threading.Lock()
        if self.csv_path and not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(_CSV_FIELDS)

    @classmethod
    def get(cls) -> DynamicTimeout:
        with cls._lock:
            if cls._instance is None:
                cls._instance = DynamicTimeout()
            return cls._instance

    # ------------------------------------------------------------------ policy
    def timeout_for(self, logical_op_id: str) -> tuple[float | None, str]:
        """Return (timeout_seconds or None, source label)."""
        if self.mode == "none":
            return None, "none"
        if self.mode != "dynamic":
            return self.fixed, "fixed"
        lat = self._lat.get(logical_op_id, [])
        if len(lat) < self.warmup:
            return (self.fixed if self.fixed is not None else self.t_max), "warmup"
        t = self.mult * float(np.quantile(lat, self.q))
        return float(min(max(t, self.t_min), self.t_max)), "dynamic"

    # --------------------------------------------------------------- recording
    def record(self, logical_op_id: str, full_op_id: str, outcome: str,
               latency: float, queue_wait: float, timeout: float | None, source: str):
        """outcome: ok | timeout | error. Only 'ok' latencies feed the estimate."""
        if outcome == "ok":
            self._lat.setdefault(logical_op_id, []).append(latency)
        if not self.csv_path:
            return
        row = [f"{time.time():.3f}", self.mode, logical_op_id, full_op_id, outcome,
               f"{latency:.3f}", f"{queue_wait:.3f}",
               "" if timeout is None else f"{timeout:.3f}", source,
               len(self._lat.get(logical_op_id, []))]
        with self._io_lock, open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
