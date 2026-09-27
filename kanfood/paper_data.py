import pandas as pd
import numpy as np
import re
from pathlib import Path
from kanfood.data import SpectralDataset, _parse_wavenumber

DATA_DIR = Path(r"C:\Users\abdulhamid batayhi\Desktop\kan2")

def _load_generic(file_path: str, sheet_name: str, header: int, 
                  target_cols: list, tagsis_col: str, name: str, brand_extractor) -> SpectralDataset:
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header)
    id_col = df.columns[0]
    df = df.dropna(subset=[id_col, target_cols[0]])
    
    meta = set([id_col, tagsis_col] + target_cols)
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
    y = df[target_cols].to_numpy(dtype=float)
    
    tagsis = pd.to_numeric(df[tagsis_col], errors="coerce").fillna(0).astype(int).to_numpy()
    
    # Extract brand
    brands = df[id_col].astype(str).apply(brand_extractor)
    # Group by brand + composition
    groups = np.array([f"B{b}_{t}" for b, t in zip(brands, y[:, 0])])
    
    return SpectralDataset(X, y, groups, tagsis, wns, target_cols, name=name)


def load_olive_oil() -> SpectralDataset:
    def ext(x):
        m = re.match(r'(\d+)', x)
        if m: return m.group(1)[0] # first digit
        if x.startswith('Z'): return x[1] # e.g. Z5 -> 5
        return "1"
        
    return _load_generic(
        DATA_DIR / "zeytinyağı tağşiş.xlsx", "Sayfa7", header=1,
        target_cols=["Zeytinyağı oranı", "Ayçiçek oranı"], tagsis_col="Tağşiş var mı",
        name="olive_oil", brand_extractor=ext
    )

def load_coffee() -> SpectralDataset:
    def ext(x):
        m = re.search(r'(\d+)\.kahve', x.lower())
        if m: return m.group(1)
        return "1"
        
    return _load_generic(
        DATA_DIR / "Kahve_Tağşiş.xlsx", "Sayfa4", header=0,
        target_cols=["Kahve oranı", "Malt unu oranı"], tagsis_col="Tağşiş var mı?",
        name="coffee", brand_extractor=ext
    )

def load_fruit_juice() -> SpectralDataset:
    def ext(x):
        m = re.search(r'p(\d+)', x.lower())
        if m: return m.group(1)
        return "1"
        
    return _load_generic(
        DATA_DIR / "meyve suyu tağşiş.xlsx", "Sayfa2", header=0,
        target_cols=["Portakal suyu oranı", "Elma suyu oranı"], tagsis_col="Tağşiş var mı",
        name="fruit_juice", brand_extractor=ext
    )
