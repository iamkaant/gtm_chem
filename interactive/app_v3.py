import argparse
import base64
import io
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update

from rdkit import Chem
from rdkit.Chem import Draw

# optional but recommended for caching artifacts
import joblib


# ---------------------------
# gtm_chem import
# ---------------------------

def import_gtm_chem(gtm_chem_path: Optional[Path]):
    if gtm_chem_path:
        p = Path(gtm_chem_path).expanduser().resolve()
        candidates = [p, p / "src", p.parent]
        added = False
        for c in candidates:
            if (c / "gtm_chem" / "__init__.py").exists():
                sys.path.insert(0, str(c))
                added = True
                break
        if not added:
            sys.path.insert(0, str(p))

    from gtm_chem import GTM, compute_fingerprints, variance_filter, pca_reduce, apply_pca
    return GTM, compute_fingerprints, variance_filter, pca_reduce, apply_pca


# ---------------------------
# Data loading
# ---------------------------

def load_library_csv(
    csv_path: Path,
    smiles_col: str,
    id_col: str,
    activity_col: str,
    dataset_name: str,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    for c in (smiles_col, id_col, activity_col):
        if c not in df.columns:
            raise ValueError(f"{csv_path.name}: missing column '{c}'. Columns: {list(df.columns)}")

    out = pd.DataFrame({
        "smiles": df[smiles_col].astype(str),
        "label": df[id_col].astype(str),
        "activity": pd.to_numeric(df[activity_col], errors="coerce"),
        "dataset": dataset_name,
    })

    # drop missing essentials
    out = out.dropna(subset=["smiles", "label"]).reset_index(drop=True)
    return out


# ---------------------------
# Landscape normalization
# ---------------------------

def normalize_landscape(land_raw) -> Dict[str, np.ndarray]:
    """
    Normalize model.landscape(...) outputs to:
      {xaxis (1D), yaxis (1D), z (2D)}
    Supports dict keys: xgrid/ygrid/z, X/Y/Z, x/y/z and tuple/list (x,y,z).
    """
    if land_raw is None:
        return {}

    if isinstance(land_raw, dict):
        d = land_raw
        x = d.get("xgrid", d.get("X", d.get("x", d.get("xx"))))
        y = d.get("ygrid", d.get("Y", d.get("y", d.get("yy"))))
        z = d.get("z", d.get("Z", d.get("zz")))
        if z is None:
            return {}
    elif isinstance(land_raw, (tuple, list)) and len(land_raw) == 3:
        x, y, z = land_raw
    else:
        return {}

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    if z.ndim != 2:
        return {}

    if x.ndim == 2 and y.ndim == 2:
        xaxis = x[0, :]
        yaxis = y[:, 0]
    else:
        xaxis = x.reshape(-1)
        yaxis = y.reshape(-1)

    if z.shape != (len(yaxis), len(xaxis)) and z.T.shape == (len(yaxis), len(xaxis)):
        z = z.T

    if z.shape != (len(yaxis), len(xaxis)):
        return {}

    return {"xaxis": xaxis, "yaxis": yaxis, "z": z}


# ---------------------------
# Map training / reuse
# ---------------------------

def train_map(
    train_smiles: list[str],
    gtm_chem_path: Optional[Path],
    n_pca: int,
    target_ratio: float,
    rbf_size: int,
    n_iter: int,
):
    GTM, compute_fingerprints, variance_filter, pca_reduce, _apply_pca = import_gtm_chem(gtm_chem_path)

    X_fps, _mask = compute_fingerprints(train_smiles, fp_type="ecfp4", return_valid_mask=True)
    X_filt, keep_mask = variance_filter(X_fps)
    X_pca, pca = pca_reduce(X_filt, n_components=n_pca)

    K_target = len(X_pca) / float(target_ratio)
    grid_target = int(max(4, np.sqrt(max(1.0, K_target))))
    model = GTM(grid_size=grid_target, rbf_size=rbf_size, n_iter=n_iter)
    model.fit_transform(X_pca)

    return model, pca, keep_mask


def project_smiles(model, pca, keep_mask, smiles: list[str], gtm_chem_path: Optional[Path]) -> np.ndarray:
    _GTM, compute_fingerprints, _variance_filter, _pca_reduce, apply_pca = import_gtm_chem(gtm_chem_path)
    X = compute_fingerprints(smiles, fp_type="ecfp4")
    Xp = apply_pca(X, pca, keep_mask=keep_mask)
    coords, _ = model.transform(Xp)
    return coords


def compute_landscape(model, pca, keep_mask, smiles_for_landscape: list[str], gtm_chem_path: Optional[Path]) -> Dict[str, np.ndarray]:
    _GTM, compute_fingerprints, _variance_filter, _pca_reduce, apply_pca = import_gtm_chem(gtm_chem_path)
    X = compute_fingerprints(smiles_for_landscape, fp_type="ecfp4")
    Xp = apply_pca(X, pca, keep_mask=keep_mask)
    land_raw = model.landscape(Xp)
    return normalize_landscape(land_raw)


def save_map(map_dir: Path, model, pca, keep_mask, landscape: Dict[str, np.ndarray]):
    map_dir.mkdir(parents=True, exist_ok=True)
    # These may or may not be serializable depending on gtm_chem implementation; joblib usually works.
    joblib.dump(model, map_dir / "gtm_model.joblib")
    joblib.dump(pca, map_dir / "pca.joblib")
    np.save(map_dir / "keep_mask.npy", keep_mask)
    if landscape:
        np.savez(map_dir / "landscape.npz", xaxis=landscape["xaxis"], yaxis=landscape["yaxis"], z=landscape["z"])


def load_map(map_dir: Path, gtm_chem_path: Optional[Path]):
    _GTM, compute_fingerprints, _variance_filter, _pca_reduce, apply_pca = import_gtm_chem(gtm_chem_path)
    model = _GTM.load(map_dir / "gtm_model.pkl")
    pca = pd.read_pickle(map_dir / "pca_model.pkl")
    keep_mask = np.load(map_dir / "var_mask.npy", allow_pickle=False)
    land_path = map_dir / "landscape.npz"
    landscape = {}
    if land_path.exists():
        d = np.load(land_path)
        landscape = {"xaxis": d["xaxis"], "yaxis": d["yaxis"], "z": d["z"]}
    return model, pca, keep_mask, landscape


# ---------------------------
# Plot / UI
# ---------------------------

def mol_png_base64(smiles: str, size=(240, 180)) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def activity_range(df: pd.DataFrame) -> Tuple[float, float]:
    v = df["activity"].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [5, 95])
    if lo == hi:
        lo, hi = float(np.min(v)), float(np.max(v))
        if lo == hi:
            hi = lo + 1.0
    return float(lo), float(hi)


def make_figures(
    df_all: pd.DataFrame,
    landscape: Dict[str, np.ndarray],
    cmin: float,
    cmax: float,
    xrange: Optional[Tuple[float, float]] = None,
    yrange: Optional[Tuple[float, float]] = None,
) -> Tuple[go.Figure, go.Figure]:

    def hm():
        if not landscape:
            return None
        return go.Heatmap(
            x=landscape["xaxis"],
            y=landscape["yaxis"],
            z=landscape["z"],
            colorscale="Greys",
            showscale=False,
            opacity=0.6,
            hoverinfo="skip",
            name="Landscape",
        )

    def scatter(sub: pd.DataFrame, show_colorbar: bool):
        custom = np.stack(
            [sub["label"].astype(str).values,
             sub["smiles"].astype(str).values,
             sub["activity"].values,
             sub["dataset"].astype(str).values],
            axis=1
        )
        return go.Scattergl(
            x=sub["x"],
            y=sub["y"],
            mode="markers",
            marker=dict(
                size=7,
                color=sub["activity"],
                cmin=cmin,
                cmax=cmax,
                colorscale="RdYlGn",
                colorbar=dict(title="activity") if show_colorbar else None,
                opacity=0.9,
                line=dict(width=0),
            ),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Dataset: %{customdata[3]}<br>"
                "Activity: %{customdata[2]:.3g}<br>"
                "<extra></extra>"
            ),
            name=sub["dataset"].iloc[0] if len(sub) else "points",
        )

    lib1 = df_all[df_all["dataset"] == "Library1"].copy()
    lib2 = df_all[df_all["dataset"] == "Library2"].copy()

    figL = go.Figure()
    t = hm()
    if t is not None: figL.add_trace(t)
    figL.add_trace(scatter(lib1, show_colorbar=True))

    figR = go.Figure()
    t = hm()
    if t is not None: figR.add_trace(t)
    figR.add_trace(scatter(lib2, show_colorbar=False))

    common = dict(
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="closest",
        dragmode="zoom",
        uirevision="keep",
        xaxis=dict(title="GTM-1"),
        yaxis=dict(title="GTM-2", scaleanchor="x", scaleratio=1),
    )
    figL.update_layout(title="Library1 (landscape + points)", **common)
    figR.update_layout(title="Library2 (landscape + points)", **common)

    if xrange is not None:
        figL.update_xaxes(range=list(xrange))
        figR.update_xaxes(range=list(xrange))
    if yrange is not None:
        figL.update_yaxes(range=list(yrange))
        figR.update_yaxes(range=list(yrange))

    return figL, figR


def extract_ranges(relayout: dict):
    if not relayout:
        return None, None
    xr = None
    yr = None
    if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
        xr = (float(relayout["xaxis.range[0]"]), float(relayout["xaxis.range[1]"]))
    if "yaxis.range[0]" in relayout and "yaxis.range[1]" in relayout:
        yr = (float(relayout["yaxis.range[0]"]), float(relayout["yaxis.range[1]"]))
    if relayout.get("xaxis.autorange") is True:
        xr = None
    if relayout.get("yaxis.autorange") is True:
        yr = None
    return xr, yr


def hover_card(hover: dict) -> html.Div:
    if not hover or "points" not in hover or not hover["points"]:
        return html.Div(
            [
                html.Div("Hover over a point to see the structure.", style={"fontSize": "14px"}),
                html.Div("Zoom/pan is linked between panels.", style={"fontSize": "12px", "opacity": 0.8}),
            ]
        )
    pt = hover["points"][0]
    label, smiles, activity, dataset = pt["customdata"][0], pt["customdata"][1], float(pt["customdata"][2]), pt["customdata"][3]
    uri = mol_png_base64(smiles)
    img = html.Img(src=uri, style={"maxWidth": "100%", "height": "auto", "borderRadius": "8px"}) if uri else html.Div(
        "Could not render structure.", style={"fontSize": "12px"}
    )
    return html.Div(
        [
            img,
            html.H4(str(label), style={"margin": "10px 0 0 0"}),
            html.Div(str(dataset), style={"fontSize": "12px", "opacity": 0.8}),
            html.Div(f"Activity: {activity:.3g}", style={"fontSize": "16px"}),
            html.Div(smiles, style={"fontFamily": "monospace", "fontSize": "12px", "opacity": 0.85, "marginTop": "6px", "wordBreak": "break-word"}),
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib1-csv", required=True)
    ap.add_argument("--lib2-csv", required=True)

    ap.add_argument("--lib1-smiles-col", required=True)
    ap.add_argument("--lib1-id-col", required=True)
    ap.add_argument("--lib1-activity-col", required=True)

    ap.add_argument("--lib2-smiles-col", required=True)
    ap.add_argument("--lib2-id-col", required=True)
    ap.add_argument("--lib2-activity-col", required=True)

    ap.add_argument("--train-on", choices=["lib1", "lib2", "both"], default="lib1",
                    help="Which library to use to train the GTM map.")
    ap.add_argument("--reuse-map", action="store_true",
                    help="Reuse existing map artifacts in --map-dir (no retraining).")
    ap.add_argument("--map-dir", default="./gtm_map_cache",
                    help="Directory to store/load map artifacts (model, PCA, keep_mask, landscape).")

    ap.add_argument("--gtm-chem-path", default=None,
                    help="Path to repo root (or src) containing gtm_chem, if not installed.")
    ap.add_argument("--n-pca", type=int, default=50)
    ap.add_argument("--target-ratio", type=float, default=8.0)
    ap.add_argument("--rbf-size", type=int, default=6)
    ap.add_argument("--n-iter", type=int, default=300)

    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    lib1 = load_library_csv(
        Path(args.lib1_csv),
        args.lib1_smiles_col,
        args.lib1_id_col,
        args.lib1_activity_col,
        dataset_name="Library1",
    )
    lib2 = load_library_csv(
        Path(args.lib2_csv),
        args.lib2_smiles_col,
        args.lib2_id_col,
        args.lib2_activity_col,
        dataset_name="Library2",
    )

    map_dir = Path(args.map_dir)
    gtm_chem_path = Path(args.gtm_chem_path).expanduser() if args.gtm_chem_path else None

    if args.reuse_map:
        model, pca, keep_mask, landscape = load_map(map_dir, gtm_chem_path)
    else:
        if args.train_on == "lib1":
            train_df = lib1
        elif args.train_on == "lib2":
            train_df = lib2
        else:
            train_df = pd.concat([lib1, lib2], ignore_index=True)

        model, pca, keep_mask = train_map(
            train_smiles=train_df["smiles"].tolist(),
            gtm_chem_path=gtm_chem_path,
            n_pca=args.n_pca,
            target_ratio=args.target_ratio,
            rbf_size=args.rbf_size,
            n_iter=args.n_iter,
        )
        landscape = compute_landscape(model, pca, keep_mask, train_df["smiles"].tolist(), gtm_chem_path)
        save_map(map_dir, model, pca, keep_mask, landscape)

    # Project both libraries onto the map
    coords1 = project_smiles(model, pca, keep_mask, lib1["smiles"].tolist(), gtm_chem_path)
    coords2 = project_smiles(model, pca, keep_mask, lib2["smiles"].tolist(), gtm_chem_path)

    lib1 = lib1.copy()
    lib2 = lib2.copy()
    lib1["x"], lib1["y"] = coords1[:, 0], coords1[:, 1]
    lib2["x"], lib2["y"] = coords2[:, 0], coords2[:, 1]

    df_all = pd.concat([lib1, lib2], ignore_index=True).dropna(subset=["x", "y", "activity"]).reset_index(drop=True)

    cmin, cmax = activity_range(df_all)
    figL, figR = make_figures(df_all, landscape, cmin, cmax)

    app = Dash(__name__)
    app.title = "GTM map explorer"

    app.layout = html.Div(
        style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"},
        children=[
            html.Div(
                style={"padding": "14px 16px 0 16px"},
                children=[
                    html.H2("GTM map explorer", style={"margin": "0 0 6px 0"}),
                    html.Div(
                        f"Train-on: {args.train_on} · Reuse-map: {args.reuse_map} · Map dir: {map_dir.resolve()}",
                        style={"fontSize": "12px", "opacity": 0.7, "marginBottom": "10px"},
                    ),
                    html.Div("Hover a point to see structure + ID + activity. Zoom is linked.", style={"opacity": 0.8}),
                ],
            ),
            html.Div(
                style={"display": "flex", "gap": "12px", "padding": "12px 16px 16px 16px", "alignItems": "stretch"},
                children=[
                    html.Div(
                        style={"flex": "1 1 0", "minWidth": "420px"},
                        children=[dcc.Graph(
                            id="graph-left",
                            figure=figL,
                            config={"displaylogo": False, "scrollZoom": True},
                            style={"height": "72vh"},
                            clear_on_unhover=False,
                        )],
                    ),
                    html.Div(
                        style={"flex": "1 1 0", "minWidth": "420px"},
                        children=[dcc.Graph(
                            id="graph-right",
                            figure=figR,
                            config={"displaylogo": False, "scrollZoom": True},
                            style={"height": "72vh"},
                            clear_on_unhover=False,
                        )],
                    ),
                    html.Div(
                        style={"width": "320px", "paddingLeft": "12px"},
                        children=[html.Div(
                            id="hover-panel",
                            style={
                                "border": "1px solid rgba(0,0,0,0.12)",
                                "borderRadius": "12px",
                                "padding": "10px",
                                "background": "white",
                                "boxShadow": "0 1px 8px rgba(0,0,0,0.06)",
                            },
                        )],
                    ),
                ],
            ),
            dcc.Store(id="range-store", data={"xrange": None, "yrange": None}),
        ],
    )

    @app.callback(
        Output("hover-panel", "children"),
        Input("graph-left", "hoverData"),
        Input("graph-right", "hoverData"),
    )
    def _hover(left_hover, right_hover):
        trig = (callback_context.triggered[0]["prop_id"] if callback_context.triggered else "")
        hover = left_hover if trig.startswith("graph-left") else right_hover
        return hover_card(hover)

    @app.callback(
        Output("graph-left", "figure"),
        Output("graph-right", "figure"),
        Output("range-store", "data"),
        Input("graph-left", "relayoutData"),
        Input("graph-right", "relayoutData"),
        State("range-store", "data"),
        prevent_initial_call=True,
    )
    def _sync(relayout_left, relayout_right, store):
        trig = (callback_context.triggered[0]["prop_id"] if callback_context.triggered else "")
        relayout = relayout_left if trig.startswith("graph-left") else relayout_right
        xr, yr = extract_ranges(relayout)

        if xr is None and yr is None and not (relayout and (relayout.get("xaxis.autorange") or relayout.get("yaxis.autorange"))):
            return no_update, no_update, no_update

        fL, fR = make_figures(df_all, landscape, cmin, cmax, xrange=xr, yrange=yr)
        return fL, fR, {"xrange": xr, "yrange": yr}

    # init hover
    app.layout.children[1].children[2].children[0].children = hover_card(None)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()