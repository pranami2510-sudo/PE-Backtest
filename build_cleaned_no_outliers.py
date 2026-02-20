"""
Build a folder like Cleaned PE Data but with outlier companies dropped and PE/price cleaning applied.
Output: Cleaned PE Data No Outliers (Date, Quarter, Close, PE) — one CSV per company.
Uses exclude_companies.txt if present; else drops the default top 3 volatile companies.
"""
import pandas as pd
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
CLEANED_DIR = _SCRIPT_DIR / "Cleaned PE Data"
OUT_DIR = _SCRIPT_DIR / "Cleaned PE Data No Outliers"
PE_CAP = 200
MAX_DAILY_RETURN = 0.20

# Default companies to drop if exclude_companies.txt is missing (top 3 by outlier days)
DEFAULT_EXCLUDE = {
    "Uttam Value Steels Ltd.",
    "F C S Software Solutions Ltd.",
    "Paras Petrofils Ltd.",
}


def load_exclude_set():
    excl_path = CLEANED_DIR.parent / "exclude_companies.txt"
    if excl_path.exists():
        names = set()
        for line in excl_path.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name and not name.startswith("#"):
                names.add(name)
        return names
    return DEFAULT_EXCLUDE


def main():
    exclude = load_exclude_set()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(CLEANED_DIR.glob("*.csv"))
    written = 0
    skipped = 0
    for i, p in enumerate(paths):
        if p.stem in exclude:
            skipped += 1
            continue
        try:
            df = pd.read_csv(p)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df["PE"] = pd.to_numeric(df["PE"], errors="coerce")
            df.loc[(df["PE"] <= 0) | (df["PE"] > PE_CAP), "PE"] = np.nan
            ret = df["Close"].pct_change()
            ret_clipped = ret.clip(lower=-MAX_DAILY_RETURN, upper=MAX_DAILY_RETURN)
            scale = (1 + ret_clipped).fillna(1).cumprod()
            df["Close"] = df["Close"].iloc[0] * scale
            if "Quarter" not in df.columns:
                df["Quarter"] = df["Date"].dt.year.astype(str) + "-Q" + df["Date"].dt.quarter.astype(str)
            out_df = df[["Date", "Quarter", "Close", "PE"]].copy()
            out_path = OUT_DIR / p.name
            out_df.to_csv(out_path, index=False)
            written += 1
        except Exception as e:
            print(f"  Skip {p.name}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(paths)}...")
    print(f"Done. Written: {written} companies to {OUT_DIR}. Skipped (excluded): {skipped}.")


if __name__ == "__main__":
    main()
