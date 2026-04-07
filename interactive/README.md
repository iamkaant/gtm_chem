
# GTM interactive map explorer (Dash)

This app reproduces the notebook's *visual exploration* as an interactive web UI:

- Two panels:
  - **Left:** GTM landscape (optional heatmap) + compound points (colored by pIC50)
  - **Right:** compound points only (same coloring)
- **Linked zoom/pan:** zoom in either panel and the other panel updates to the same view.
- **Hover card:** hover any point to see the compound **structure**, **label**, and **pIC50**.

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Provide data

Place files in `data/` (or another folder you pass via `--data-dir`):

### Required
- `data/points.parquet` (or `data/points.csv`) with columns:
  - `x`, `y` (GTM coordinates)
  - `label` (compound ID)
  - `smiles`
  - `pIC50`

### Optional
- `data/landscape.npz` with arrays: `xgrid`, `ygrid`, `z`
  - used as a grayscale background "map" in the left panel

### Converting existing outputs
If you already have a dataframe with these columns, you can export using:

```bash
python prepare_data.py --infile your_points.csv --out-dir data
```

## 3) Run

```bash
python app.py --data-dir data --port 8050
```

Then open http://127.0.0.1:8050

## Notes for adapting from your notebook

Your notebook currently builds coordinates like:

- `coords_chembl` (N x 2)
- `coords_seth` (M x 2)
- `chembl_smi`, `seth_smi`
- `chembl_ic50s`, `seth_ic50s` (your pChEMBL / pKi columns)

To feed the app, you just need a single table:

| x | y | label | smiles | pIC50 | dataset |
|---|---|-------|--------|-------|---------|

You can concatenate multiple datasets and include a `dataset` column; the app keeps extra metadata columns and you can extend the hover card to show them.
