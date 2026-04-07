
"""
prepare_data.py

A small "bridge" script to convert the notebook's outputs into app-friendly files.

Two supported modes:

A) If you already have the notebook outputs as arrays/dataframes (coords, pIC50, labels, smiles, and landscape grid),
   you can adapt the `export_from_objects(...)` function and call it from your analysis pipeline.

B) If you want to run the notebook itself and capture variables, you *can* do that via papermill or nbclient,
   but it usually requires your local data paths to exist. This script intentionally stays lightweight and
   focuses on the export format expected by app.py.

Output files written to <out_dir>/:
  - points.parquet
  - landscape.npz (optional)

Expected point columns:
  x, y, label, smiles, pIC50
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def export_from_dataframe(
    df: pd.DataFrame,
    out_dir: Path,
    x_col: str = "x",
    y_col: str = "y",
    label_col: str = "label",
    smiles_col: str = "smiles",
    p_col: str = "pIC50",
    landscape_npz: Optional[Path] = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize column names into the app schema
    out = pd.DataFrame(
        {
            "x": df[x_col].astype(float),
            "y": df[y_col].astype(float),
            "label": df[label_col].astype(str),
            "smiles": df[smiles_col].astype(str),
            "pIC50": pd.to_numeric(df[p_col]),
        }
    )

    # Keep any extra metadata columns (optional)
    extra_cols = [c for c in df.columns if c not in {x_col, y_col, label_col, smiles_col, p_col}]
    for c in extra_cols:
        out[c] = df[c]

    out.to_parquet(out_dir / "points.parquet", index=False)

    if landscape_npz is not None:
        # Expect xgrid, ygrid, z
        d = np.load(landscape_npz)
        for k in ("xgrid", "ygrid", "z"):
            if k not in d:
                raise ValueError(f"{landscape_npz} missing array: {k}")
        np.savez(out_dir / "landscape.npz", xgrid=d["xgrid"], ygrid=d["ygrid"], z=d["z"])

    print(f"Wrote: {(out_dir / 'points.parquet').resolve()}")
    if landscape_npz is not None:
        print(f"Wrote: {(out_dir / 'landscape.npz').resolve()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="CSV/Parquet file with coords + smiles + pIC50")
    ap.add_argument("--out-dir", default="data", help="Output directory for app files")
    ap.add_argument("--x-col", default="x")
    ap.add_argument("--y-col", default="y")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--p-col", default="pIC50")
    ap.add_argument("--landscape-npz", default=None, help="Optional NPZ with xgrid,ygrid,z")
    args = ap.parse_args()

    infile = Path(args.infile)
    out_dir = Path(args.out_dir)
    if infile.suffix.lower() == ".parquet":
        df = pd.read_parquet(infile)
    else:
        df = pd.read_csv(infile)

    export_from_dataframe(
        df=df,
        out_dir=out_dir,
        x_col=args.x_col,
        y_col=args.y_col,
        label_col=args.label_col,
        smiles_col=args.smiles_col,
        p_col=args.p_col,
        landscape_npz=Path(args.landscape_npz) if args.landscape_npz else None,
    )


if __name__ == "__main__":
    main()
