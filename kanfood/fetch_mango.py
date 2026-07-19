"""Reproducibly fetch the mango DMC NIR dataset (Anderson et al. 2020, Mendeley 46htwnp833) used by
load_mango(). Downloads the raw release mirror from the spectral-datasets GitHub and extracts the single
CSV to data/mango/raw/. The data (~30 MB) is not committed; run this once. Run: python -m kanfood.fetch_mango"""
import sys
import io
import zipfile
import urllib.request
from pathlib import Path

RAW_ZIP_URL = "https://github.com/spectral-datasets/mango-dmc/releases/download/raw-v1/46htwnp833-1.zip"
from kanfood.data import MANGO_PATH
EXPECTED = Path(MANGO_PATH)
DEST = EXPECTED.parent


def main():
    if EXPECTED.exists():
        print(f"Already present: {EXPECTED} ({EXPECTED.stat().st_size/1e6:.1f} MB)")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {RAW_ZIP_URL} ...")
    with urllib.request.urlopen(RAW_ZIP_URL) as r:
        data = r.read()
    print(f"  got {len(data)/1e6:.1f} MB; extracting...")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(DEST)
    if EXPECTED.exists():
        print(f"OK -> {EXPECTED} ({EXPECTED.stat().st_size/1e6:.1f} MB)")
    else:
        print("Extracted, but expected CSV not found; contents:", [p.name for p in DEST.iterdir()])
        sys.exit(1)


if __name__ == "__main__":
    main()
