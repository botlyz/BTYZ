"""Rolling-window train/test split for Walk-Forward Analysis."""
from typing import List, Tuple

from .config import BAR_MINUTES


def compute_folds(
    n_bars: int,
    train_days: int,
    test_days: int,
    step_days: int,
    tf: str,
    min_train_bars: int = 500,
    min_test_bars: int = 100,
) -> List[Tuple[int, int, int]]:
    """Return (train_start, train_end, test_end) tuples for each fold.

    step_days < test_days → overlapping folds.
    """
    bar_min = BAR_MINUTES.get(tf, 15)
    train_bars = int(train_days * 1440 / bar_min)
    test_bars = int(test_days * 1440 / bar_min)
    step_bars = int(step_days * 1440 / bar_min)

    folds = []
    i = 0
    while True:
        s = i * step_bars
        train_end = s + train_bars
        test_end = train_end + test_bars
        if test_end > n_bars:
            break
        if (train_end - s) >= min_train_bars and (test_end - train_end) >= min_test_bars:
            folds.append((s, train_end, test_end))
        i += 1
    return folds
