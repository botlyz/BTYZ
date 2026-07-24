"""Fan-out standard quantlab : saturer les 24 threads sans fuite de RAM.

Règles :
- un seul point d'entrée (`pool_map`) pour paralléliser paires / folds /
  permutations / nulles — pas de ProcessPoolExecutor ad hoc dans les modules ;
- workers recyclés tous les MAX_TASKS_PER_CHILD tâches (la RAM d'un worker
  repart de zéro, donc pas de montée continue) ;
- gc + purge des caches data à la fin de chaque tâche lourde côté worker.
"""
from __future__ import annotations

import gc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Optional

from quantlab.config import MAX_TASKS_PER_CHILD, N_WORKERS


def _worker_cleanup() -> None:
    """À appeler en fin de tâche lourde dans un worker : libère caches + gc."""
    try:
        from engine.data_loader import clear_cache
        clear_cache()
    except Exception:
        pass
    gc.collect()


def _wrap(fn: Callable, cleanup_every: int, args: tuple) -> Any:
    res = fn(*args) if isinstance(args, tuple) else fn(args)
    # le recyclage max_tasks_per_child fait le gros du travail ; on aide entre-temps
    if cleanup_every and (os.getpid() + id(args)) % cleanup_every == 0:
        _worker_cleanup()
    return res


def pool_map(fn: Callable, items: Iterable, *, workers: Optional[int] = None,
             max_tasks_per_child: Optional[int] = MAX_TASKS_PER_CHILD,
             progress_handle=None, ordered: bool = True,
             initializer: Optional[Callable] = None,
             initargs: tuple = ()) -> list:
    """map parallèle process-based.

    - fn(item) picklable (utiliser quantlab.registry pour ré-importer une
      stratégie côté worker).
    - workers: défaut config.N_WORKERS (=22 sur cette machine).
    - max_tasks_per_child: recyclage anti-fuite (None pour désactiver, ex.
      tâches très courtes où le respawn coûterait plus que la RAM).
    - progress_handle: TaskHandle quantlab.progress -> advance(1) par tâche.
    - ordered=False: résultats dans l'ordre d'achèvement (plus fluide pour
      la barre de progression, à utiliser quand l'ordre n'importe pas).
    """
    items = list(items)
    if not items:
        return []
    workers = min(workers or N_WORKERS, len(items))
    if workers <= 1:
        out = []
        for it in items:
            out.append(fn(it))
            if progress_handle is not None:
                progress_handle.advance(1)
        return out

    exe = ProcessPoolExecutor(
        max_workers=workers,
        max_tasks_per_child=max_tasks_per_child,
        initializer=initializer,
        initargs=initargs,
    )
    try:
        futures = {exe.submit(fn, it): idx for idx, it in enumerate(items)}
        if ordered:
            results: list = [None] * len(items)
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
                if progress_handle is not None:
                    progress_handle.advance(1)
            return results
        results = []
        for fut in as_completed(futures):
            results.append(fut.result())
            if progress_handle is not None:
                progress_handle.advance(1)
        return results
    finally:
        exe.shutdown(wait=True, cancel_futures=True)
        gc.collect()


def chunked(seq: list, n_chunks: int) -> list[list]:
    """Découpe équilibrée pour amortir le coût de spawn sur tâches courtes."""
    n_chunks = max(1, min(n_chunks, len(seq)))
    k, m = divmod(len(seq), n_chunks)
    out, start = [], 0
    for i in range(n_chunks):
        size = k + (1 if i < m else 0)
        out.append(seq[start:start + size])
        start += size
    return out
