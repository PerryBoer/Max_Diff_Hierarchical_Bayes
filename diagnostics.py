# hb_diag_and_export.py
# Post-run diagnostics & Sawtooth-style exports for MaxDiff HB (open-source, Pythonic).
# Focus: loading, plotting, utilities/shares export, per-respondent RLH. No sampler code here.

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === Validated likelihood utilities (yours) ===
from hb_likelihood import expand_effects, loglik_respondent


# ---------- Small adapter around your (snapshot) spec ----------
@dataclass(frozen=True)
class SpecLite:
    J: int
    N: int
    E: int
    item_ids: list
    respondent_ids: list
    respondent_idx: np.ndarray
    stage: np.ndarray
    set_ptr: np.ndarray
    set_items: np.ndarray
    chosen_item: np.ndarray
    weight: np.ndarray

def _load_spec_from_snapshot(snapshot_path: Path) -> SpecLite:
    z = np.load(snapshot_path, allow_pickle=True)

    # robust pulls + fallbacks
    J = int(z["J"][0]) if "J" in z.files else int(np.max(z["set_items"])) + 1
    respondent_ids = [x for x in z["respondent_ids"]] if "respondent_ids" in z.files else []
    N = int(z["N"][0]) if "N" in z.files else len(respondent_ids)
    E = int(z["E"][0]) if "E" in z.files else int(z["respondent_idx"].shape[0])

    return SpecLite(
        J=J,
        N=N,
        E=E,
        item_ids=[int(x) for x in z["item_ids"]],
        respondent_ids=respondent_ids if respondent_ids else list(range(N)),
        respondent_idx=z["respondent_idx"],
        stage=z["stage"],
        set_ptr=z["set_ptr"],
        set_items=z["set_items"],
        chosen_item=z["chosen_item"],
        weight=z["weight"],
    )

def get_J(spec: SpecLite) -> int:
    return int(spec.J)

def get_item_ids(spec: SpecLite) -> Iterable:
    return list(spec.item_ids)

def get_respondent_ids(spec: SpecLite) -> Iterable:
    return list(spec.respondent_ids)


# ---------- Data containers ----------
@dataclass(frozen=True)
class HBResults:
    beta_mean: np.ndarray         # shape (N, p)  -- p == J-1 with effects-coding
    alpha_mean: np.ndarray        # shape (p,)
    D_mean: np.ndarray            # shape (p, p)
    traces: Dict[str, np.ndarray] # each 1D over iterations
    beta_draws: Optional[np.ndarray] = None  # optional (n_keep, N, p)
    alpha_draws: Optional[np.ndarray] = None # optional (n_keep, p)
    D_draws: Optional[np.ndarray] = None     # optional (n_keep, p, p)

@dataclass(frozen=True)
class Bundle:
    results: HBResults
    meta: Dict
    spec: SpecLite


# ---------- Bundle loader (snapshot-based; no pickle) ----------
def load_bundle(
    bundle_dir: Path,
    npz_name: str = "hb_results_bundle.npz",
    meta_name: str = "hb_results_metadata.json",
    snapshot_name: str = "hb_results_spec_snapshot.npz",
) -> Bundle:
    bundle_dir = Path(bundle_dir)
    npz_path = bundle_dir / npz_name
    meta_path = bundle_dir / meta_name
    snap_path = bundle_dir / snapshot_name

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    if not snap_path.exists():
        raise FileNotFoundError(f"Missing {snap_path} (spec snapshot).")

    with np.load(npz_path, allow_pickle=True) as z:
        beta_mean = z["beta_mean"]
        alpha_mean = z["alpha_mean"]
        D_mean = z["D_mean"]
        traces = {k: z[k] for k in z.files if k.startswith("trace_")}
        beta_draws = z["beta_draws"] if "beta_draws" in z.files else None
        alpha_draws = z["alpha_draws"] if "alpha_draws" in z.files else None
        D_draws = z["D_draws"] if "D_draws" in z.files else None

    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    spec = _load_spec_from_snapshot(snap_path)

    results = HBResults(
        beta_mean=beta_mean,
        alpha_mean=alpha_mean,
        D_mean=D_mean,
        traces=traces,
        beta_draws=beta_draws,
        alpha_draws=alpha_draws,
        D_draws=D_draws,
    )
    return Bundle(results=results, meta=meta, spec=spec)


# ---------- Diagnostics ----------
def tail_slice(x: np.ndarray, tail_frac: float) -> np.ndarray:
    tail_frac = float(tail_frac)
    if not (0 < tail_frac <= 1):
        raise ValueError("tail_frac must be in (0, 1].")
    n = x.shape[0]
    start = max(0, n - int(math.ceil(tail_frac * n)))
    return x[start:]

def slope_last_tail(x: np.ndarray, tail_frac: float) -> float:
    """Return OLS slope over last tail fraction of the series."""
    y = tail_slice(x, tail_frac)
    t = np.arange(y.shape[0], dtype=float)
    if y.size < 2:
        return np.nan
    return float(np.polyfit(t, y, 1)[0])  # degree 1: slope

def summarize_traces(
    traces: Dict[str, np.ndarray],
    tail_frac: float = 0.5
) -> pd.DataFrame:
    rows = []
    for name, arr in traces.items():
        tail = tail_slice(arr, tail_frac)
        rows.append(
            dict(
                metric=name.replace("trace_", ""),
                last=float(arr[-1]),
                mean_tail=float(np.mean(tail)),
                sd_tail=float(np.std(tail, ddof=1)) if tail.size > 1 else np.nan,
                slope_tail=slope_last_tail(arr, tail_frac),
                n_iters=int(arr.shape[0]),
            )
        )
    return pd.DataFrame(rows).set_index("metric")

def plot_traces(
    traces: Dict[str, np.ndarray],
    outdir: Path,
    tail_frac: float = 0.5,
    style: Optional[str] = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if style:
        plt.style.use(style)

    for name, arr in traces.items():
        metric = name.replace("trace_", "")
        fig = plt.figure(figsize=(8, 4.5))
        ax = plt.gca()

        ax.plot(np.arange(arr.shape[0]), arr, lw=1)
        # Mark tail window
        n = arr.shape[0]
        start = max(0, n - int(math.ceil(tail_frac * n)))
        ax.axvspan(start, n, alpha=0.08, label=f"tail {tail_frac:.0%}")

        # Slope over tail
        sl = slope_last_tail(arr, tail_frac)
        ax.set_title(f"{metric} | tail-slope={sl: .3g}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(metric)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / f"trace_{metric}.png", dpi=150)
        plt.close(fig)


# ---------- Utilities & Shares ----------
def expand_beta_tilde_all(beta_tilde_all: np.ndarray, J: int) -> np.ndarray:
    """Row-wise expand effects-coded betas to full J utilities."""
    N, p = beta_tilde_all.shape
    if p != J - 1:
        raise ValueError(f"Expected p=J-1 (got p={p}, J={J})")
    full = np.empty((N, J), dtype=float)
    for i in range(N):
        full[i, :] = expand_effects(beta_tilde_all[i], J)
    return full

def zero_center_rows(U: np.ndarray) -> np.ndarray:
    """Subtract row means (per respondent) to get zero-centered utilities."""
    return U - U.mean(axis=1, keepdims=True)

def softmax_rows(U: np.ndarray) -> np.ndarray:
    """Row-wise softmax (numerically stable). Returns probabilities summing to 1."""
    M = U.max(axis=1, keepdims=True)
    e = np.exp(U - M)
    denom = e.sum(axis=1, keepdims=True)
    return e / denom


# ---------- RLH ----------
def rlh_per_respondent_from_likelihood(
    beta_tilde_all: np.ndarray,
    spec: SpecLite
) -> pd.DataFrame:
    """
    RLH_i = geometric mean of predicted choice probabilities across events for respondent i.
    Implemented via RLH_i = exp(loglik_i / E_i).
    """
    N, _ = beta_tilde_all.shape
    rids = get_respondent_ids(spec)
    if len(rids) != N:
        raise ValueError(f"Spec/respondent mismatch: N={N} vs {len(rids)}")

    rlhs, logliks, n_events = [], [], []
    for i in range(N):
        ll_i = float(loglik_respondent(beta_tilde_all[i], spec, i))
        # number of events for respondent i (integer-coded in respondent_idx)
        E_i = int(np.sum(spec.respondent_idx == i))
        rlhs.append(math.exp(ll_i / E_i))
        logliks.append(ll_i)
        n_events.append(E_i)

    df = pd.DataFrame(dict(
        respondent_id=rids,
        loglik=logliks,
        n_events=n_events,
        rlh=rlhs,
    ))
    return df


# ---------- Exports ----------
def write_matrix_csv(
    M: np.ndarray,
    row_ids: Iterable,
    col_ids: Iterable,
    path: Path
) -> None:
    df = pd.DataFrame(M, index=row_ids, columns=col_ids)
    df.index.name = "respondent_id"
    df.to_csv(path, float_format="%.6f", encoding="utf-8")


# ---------- High-level pipeline ----------
def run_diag_and_export(
    bundle_dir: Path,
    outdir: Path,
    tail_frac: float = 0.5,
    style: Optional[str] = None,
    npz_name: str = "hb_results_bundle.npz",
    meta_name: str = "hb_results_metadata.json",
    snapshot_name: str = "hb_results_spec_snapshot.npz",
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(
        bundle_dir=bundle_dir,
        npz_name=npz_name,
        meta_name=meta_name,
        snapshot_name=snapshot_name,
    )
    results, spec = bundle.results, bundle.spec
    J = get_J(spec)
    item_ids = get_item_ids(spec)
    respondent_ids = get_respondent_ids(spec)

    # 1) Diagnostics: summarize & plot traces
    traces = results.traces
    summary = summarize_traces(traces, tail_frac=tail_frac)
    summary.to_csv(outdir / "trace_summary_tail.csv", float_format="%.6f", encoding="utf-8")
    plot_traces(traces, outdir=outdir / "traces", tail_frac=tail_frac, style=style)

    # 2) Expand betas to full-J utilities, zero-center, shares
    beta_full = expand_beta_tilde_all(results.beta_mean, J)
    utils_zc = zero_center_rows(beta_full)
    shares = softmax_rows(utils_zc) * 100.0  # Sawtooth-style percentage scores that sum to 100

    # 3) Per-respondent RLH
    rlh_df = rlh_per_respondent_from_likelihood(results.beta_mean, spec)
    rlh_df.to_csv(outdir / "hb_rlh.csv", index=False, float_format="%.6f", encoding="utf-8")

    # 4) Exports
    write_matrix_csv(utils_zc, respondent_ids, item_ids, outdir / "hb_utilities.csv")
    write_matrix_csv(shares, respondent_ids, item_ids, outdir / "hb_shares.csv")

    # 5) Save a compact run report
    report = {
        "bundle_dir": str(Path(bundle_dir).resolve()),
        "outdir": str(Path(outdir).resolve()),
        "J": J,
        "N": int(beta_full.shape[0]),
        "trace_metrics": list(traces.keys()),
        "tail_frac": tail_frac,
        "files": {
            "trace_summary_tail": "trace_summary_tail.csv",
            "traces_png_dir": "traces/",
            "hb_utilities": "hb_utilities.csv",
            "hb_shares": "hb_shares.csv",
            "hb_rlh": "hb_rlh.csv",
        },
        "notes": (
            "RLH is the geometric mean of predicted choice probabilities per respondent; "
            "PctCert and RLH are standard HB convergence/fit diagnostics. "
            "Utilities are zero-centered per respondent; shares sum to 100 per respondent."
        ),
    }
    with open(outdir / "run_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(
        description="Diagnostics & Sawtooth-style exports for MaxDiff HB results."
    )
    p.add_argument("--bundle-dir", type=Path, default=Path("."))   # was required=True
    p.add_argument("--outdir", type=Path, default=Path("./diag"))  # was required=True
    p.add_argument("--tail-frac", type=float, default=0.5,
                   help="Tail fraction of iterations for summaries/slopes (default: 0.5).")
    p.add_argument("--mpl-style", type=str, default=None,
                   help="Optional Matplotlib style (e.g., 'ggplot').")
    p.add_argument("--npz-name", type=str, default="hb_results_bundle.npz")
    p.add_argument("--meta-name", type=str, default="hb_results_metadata.json")
    p.add_argument("--snapshot-name", type=str, default="hb_results_spec_snapshot.npz")
    args = p.parse_args()

    run_diag_and_export(
        bundle_dir=args.bundle_dir,
        outdir=args.outdir,
        tail_frac=args.tail_frac,
        style=args.mpl_style,
        npz_name=args.npz_name,
        meta_name=args.meta_name,
        snapshot_name=args.snapshot_name,
    )

if __name__ == "__main__":
    main()
