from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd

from kanfood._paths import data_path, first_existing

# Open dataset ships under data/tahini/; also resolves a repo-root copy for backward compatibility.
TAHINI_PATH = first_existing("KANFOOD_TAHINI_PATH",
                             "data/tahini/Tahin_Aycicek1.xlsx", "Tahin_Aycicek1.xlsx")
TARGET_COLS = ["Tahin Oranı", "Fıstık Ezmesi Oranı", "Ayçiçek ezmesi oranı"]
TAGSIS_COL = "Tağşiş Var mı"
ID_COL = "isim"


@dataclass
class SpectralDataset:
    X: np.ndarray              # (n_samples, n_channels), columns sorted ascending (cm-1 for FTIR, nm for NIR)
    y: np.ndarray              # (n_samples, n_targets); composition (sums to 100) or single continuous target
    groups: np.ndarray         # (n_samples,) physical-sample id (for group splitting)
    tagsis: np.ndarray         # (n_samples,) 1=adulterated, 0=pure (0 throughout for pure-regression datasets)
    wavenumbers: np.ndarray    # (n_channels,) ascending axis
    target_names: List[str]
    name: str = "dataset"
    sets: Optional[np.ndarray] = None   # optional predefined split labels (e.g. mango 'Cal'/'Tuning'/'Val Ext')


def _parse_wavenumber(col) -> Optional[float]:
    try:
        return float(str(col).replace(",", "."))
    except (ValueError, TypeError):
        return None


def load_tahini(path: str = TAHINI_PATH) -> SpectralDataset:
    df = pd.read_excel(path, sheet_name=0)
    meta = {ID_COL, TAGSIS_COL, *TARGET_COLS}
    wl_cols, wns = [], []
    for c in df.columns:
        if c in meta:
            continue
        wn = _parse_wavenumber(c)
        if wn is not None:
            wl_cols.append(c)
            wns.append(wn)
    wns = np.asarray(wns, dtype=float)
    order = np.argsort(wns)
    wns = wns[order]
    X = df[wl_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[:, order]
    y = df[TARGET_COLS].to_numpy(dtype=float)
    groups = df[ID_COL].astype(str).to_numpy()
    tagsis = pd.to_numeric(df[TAGSIS_COL], errors="coerce").fillna(0).astype(int).to_numpy()
    return SpectralDataset(X, y, groups, tagsis, wns, list(TARGET_COLS), name="tahini")


OILS_PATH = data_path("KANFOOD_OILS_PATH", "data", "oils", "ATRAdulteration3.csv")
OILS_TARGETS = ["EVOO", "Almond", "Canola", "Corn"]   # EVOO = authentic primary fraction


def load_oils(path: str = OILS_PATH) -> SpectralDataset:
    """EVOO adulterated with almond/canola/corn (ATR-FTIR, cm-1). y[:,0] = EVOO% (authentic fraction).
    Replicate scans share a blend composition -> group by composition for leakage-free splits."""
    df = pd.read_csv(path)
    meta = set(OILS_TARGETS)
    wl_cols, wns = [], []
    for c in df.columns:
        if c in meta:
            continue
        wn = _parse_wavenumber(c)
        if wn is not None:
            wl_cols.append(c)
            wns.append(wn)
    wns = np.asarray(wns, dtype=float)
    order = np.argsort(wns)
    wns = wns[order]
    X = df[wl_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[:, order]
    y = df[OILS_TARGETS].to_numpy(dtype=float)
    groups = df[OILS_TARGETS].round().astype(int).astype(str).agg("-".join, axis=1).to_numpy()
    tagsis = (y[:, 0] < 100).astype(int)
    return SpectralDataset(X, y, groups, tagsis, wns, list(OILS_TARGETS), name="oils")


MANGO_PATH = data_path("KANFOOD_MANGO_PATH", "data", "mango", "raw", "NAnderson2020MendeleyMangoNIRData.csv")
MANGO_META = ["Set", "Season", "Region", "Date", "Type", "Cultivar", "Pop", "Temp", "DM"]


def load_mango(path: str = MANGO_PATH) -> SpectralDataset:
    """Mango dry-matter-content (DMC) NIR benchmark (Anderson et al. 2020, Mendeley 46htwnp833):
    11,691 spectra x 306 channels (285-1200 nm), single continuous target = DM (%). Large and
    genuinely nonlinear -- the canonical 'CNN beats PLS' NIR dataset. groups = Pop (112 populations,
    leakage-safe; replicate scans share a Pop). `sets` = the published interseason split
    ('Cal' / 'Tuning' / 'Val Ext'). tagsis is all 0 (pure quantification, no adulteration label).
    Single target -> models skip composition normalization (see models._Base)."""
    df = pd.read_csv(path)
    meta = set(MANGO_META)
    wl_cols, wns = [], []
    for c in df.columns:
        if c in meta:
            continue
        wn = _parse_wavenumber(c)
        if wn is not None:
            wl_cols.append(c)
            wns.append(wn)
    wns = np.asarray(wns, dtype=float)
    order = np.argsort(wns)
    wns = wns[order]
    X = df[wl_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[:, order]
    y = df["DM"].to_numpy(dtype=float).reshape(-1, 1)
    groups = df["Pop"].astype(str).to_numpy()
    sets = df["Set"].astype(str).to_numpy()
    tagsis = np.zeros(len(df), dtype=int)
    return SpectralDataset(X, y, groups, tagsis, wns, ["DM"], name="mango", sets=sets)


def resample_spectra(X: np.ndarray, wn_src: np.ndarray, wn_dst: np.ndarray) -> np.ndarray:
    """Linear-interpolate each spectrum from its native cm-1 axis onto a common grid (for transfer)."""
    out = np.empty((X.shape[0], len(wn_dst)), dtype=float)
    for i in range(X.shape[0]):
        out[i] = np.interp(wn_dst, wn_src, X[i])
    return out


def common_grid(ds_a: "SpectralDataset", ds_b: "SpectralDataset", step: float = 2.0) -> np.ndarray:
    """Overlapping cm-1 range of two datasets at a fixed step (for cross-food transfer)."""
    lo = max(ds_a.wavenumbers.min(), ds_b.wavenumbers.min())
    hi = min(ds_a.wavenumbers.max(), ds_b.wavenumbers.max())
    return np.arange(np.ceil(lo), np.floor(hi), step)
