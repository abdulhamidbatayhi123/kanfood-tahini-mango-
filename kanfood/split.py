from typing import List, Tuple
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def group_holdout_split(groups, test_size=0.30, seed=42) -> Tuple[np.ndarray, np.ndarray]:
    """Train/test indices with NO shared group (isim) between sets."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(np.zeros(len(groups)), groups=groups))
    return tr, te


def stratified_group_kfold(groups, y_class, n_splits=5, seed=42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Stratify by class while keeping each group entirely within one fold."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(groups)), y_class, groups))
