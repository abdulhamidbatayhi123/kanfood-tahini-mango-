import numpy as np
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             f1_score, accuracy_score, precision_score, recall_score)


def normalize_to_100(pred: np.ndarray) -> np.ndarray:
    pred = np.clip(pred, 0, None)
    s = pred.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return pred / s * 100.0


def rmse(yt, yp) -> float:
    return float(np.sqrt(mean_squared_error(yt, yp)))


def rpd(yt, yp) -> float:
    e = rmse(yt, yp)
    return float(np.std(yt) / e) if e > 0 else float("inf")


def regression_metrics(y_true, y_pred, target_names):
    m = {}
    for i, name in enumerate(target_names):
        m[f"R2_{name}"] = r2_score(y_true[:, i], y_pred[:, i])
        m[f"MAE_{name}"] = mean_absolute_error(y_true[:, i], y_pred[:, i])
        m[f"RMSE_{name}"] = rmse(y_true[:, i], y_pred[:, i])
        m[f"RPD_{name}"] = rpd(y_true[:, i], y_pred[:, i])
    for agg in ("R2", "MAE", "RMSE", "RPD"):
        m[f"{agg}_mean"] = float(np.mean([m[f"{agg}_{n}"] for n in target_names]))
    return m


def classification_from_tahini(pred_tahini, threshold):
    return (np.asarray(pred_tahini) < threshold).astype(int)


def classification_metrics(t_true, t_pred):
    return {
        "Acc": accuracy_score(t_true, t_pred),
        "F1": f1_score(t_true, t_pred, zero_division=0),
        "Precision": precision_score(t_true, t_pred, zero_division=0),
        "Recall": recall_score(t_true, t_pred, zero_division=0),
    }


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, seed=42, conf=0.95, groups=None):
    """CI for a metric. NOTE: caller MUST pass the SAME quantity for point estimate and CI
    (this avoids the legacy bug where a 3-target mean was paired with a 1-target CI).

    When `groups` is given, a CLUSTER (group) bootstrap resamples whole groups (isim / Pop)
    rather than individual scans (rigor audit 1.2). Test rows are replicate scans clustered in
    few physical samples / populations; resampling scans as if independent makes the CI
    artificially tight, so the grouped resample yields an honest (wider) interval."""
    rng = np.random.RandomState(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = np.empty(n_boot)
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
        ng = len(uniq)
        for b in range(n_boot):
            chosen = rng.choice(uniq, size=ng, replace=True)
            idx = np.concatenate([idx_by_group[g] for g in chosen])
            scores[b] = metric_fn(y_true[idx], y_pred[idx])
    else:
        n = len(y_true)
        for b in range(n_boot):
            idx = rng.randint(0, n, n)
            scores[b] = metric_fn(y_true[idx], y_pred[idx])
    a = (1 - conf) / 2
    return (float(scores.mean()),
            float(np.percentile(scores, a * 100)),
            float(np.percentile(scores, (1 - a) * 100)))


def corrected_resampled_ttest(scores_a, scores_b, rho=None):
    """Nadeau-Bengio corrected paired t-test for comparing two models across CV folds
    (rigor audit 2.1). A plain paired t-test on overlapping CV folds is optimistic because the
    fold train sets overlap; the correction inflates the variance by (1/k + rho), rho = n_test/
    n_train (default 1/(k-1) for k-fold CV). Returns (t, p) two-sided with df = k-1."""
    from scipy import stats as _st
    d = np.asarray(scores_a, float) - np.asarray(scores_b, float)
    k = len(d)
    if rho is None:
        rho = 1.0 / (k - 1)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt(var * (1.0 / k + rho))
    return float(t), float(2 * _st.t.sf(abs(t), df=k - 1))


def holm_bonferroni(pvals):
    """Holm-Bonferroni step-down adjusted p-values for a family of tests (rigor audit 2.1).
    Returns adjusted p-values in the original order; compare against alpha as usual."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)          # enforce monotonicity
        adj[i] = min(running, 1.0)
    return adj.tolist()
