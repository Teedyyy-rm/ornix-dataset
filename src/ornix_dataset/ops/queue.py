"""Bounded, stage-separated worker executor (spec §8).

Resource classes (io/cpu/ml) get independent bounded pools so an expensive ML
stage never starves I/O and GPU work only runs after cheaper CPU/rights gates
pass (enforced by running stages in order). Deterministic, ordered results.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional


class StageExecutor:
    def __init__(self, io_workers: int = 8, cpu_workers: int = 4, ml_workers: int = 1):
        self.limits = {"io": io_workers, "cpu": cpu_workers, "ml": ml_workers}

    def map_stage(self, resource: str, fn: Callable[[Any], Any], items: Iterable[Any],
                  max_workers: Optional[int] = None) -> List[Any]:
        items = list(items)
        workers = max(1, min(max_workers or self.limits.get(resource, 4), len(items) or 1))
        results: List[Any] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
            for fut in futs:
                idx = futs[fut]
                results[idx] = fut.result()
        return results
