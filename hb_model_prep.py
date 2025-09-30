# hb_model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal, List, Dict, Hashable, Optional, Tuple
import numpy as np

# ---- 1) Public spec the sampler will consume --------------------------------

@dataclass(frozen=True)
class HBModelSpec:
    # Universe & coding
    J: int
    p: int                      # = J-1 under effects coding
    item_ids: List[int]
    coding: Literal["effects"]  # we start with effects only

    # Event structure (from PreparedMaxDiff)
    N: int
    E: int
    respondent_idx: np.ndarray      # (E,), int32
    stage: np.ndarray               # (E,), uint8, 0=Best, 1=Worst
    set_ptr: np.ndarray             # (E+1,), int32
    set_items: np.ndarray           # (set_ptr[-1],), int32 in [0, J-1]
    chosen_item: np.ndarray         # (E,), int32 in [0, J-1]
    weight: np.ndarray              # (E,), float32

    # Priors (Sawtooth-style, effects coding)
    v_prior: float                  # variance scale hyperparameter
    df_add: int                     # IW df = p + df_add
    iw_df: int                      # computed = p + df_add
    beta_struct_prior_cov: np.ndarray  # (p,p), effects-structured prior cov
    iw_scale: np.ndarray               # (p,p), IW scale

    # Hyper-prior on alpha (we keep simple, CBC/HB-compatible)
    alpha_prior_tau: float          # store τ for alpha ~ N(0, τ I); used rarely

    # Utilities
    effects_expand: Callable[[np.ndarray], np.ndarray]  # β̃ (p,) -> β (J,) s.t. sum β = 0

    # MCMC knobs (sampler will read these; harmless here)
    burn_in: int
    iters: int
    thin: int
    mh_step_init: float
    target_accept: float


# ---- 2) Builder from PreparedMaxDiff ----------------------------------------

class HBModelSpecBuilder:
    @staticmethod
    def build_from_prepared(
        prep,                           # your PreparedMaxDiff instance
        *,
        coding: Literal["effects"] = "effects",
        v_prior: float = 2.0,           # Sawtooth Appendix C uses ~2.0 as default
        df_add: int = 5,                # IW df = p + df_add
        alpha_prior_tau: float = 1e6,   # weakly-informative α prior variance
        burn_in: int = 10_000,
        iters: int = 20_000,
        thin: int = 10,
        mh_step_init: float = 0.10,
        target_accept: float = 0.30,
    ) -> HBModelSpec:
        if coding != "effects":
            raise NotImplementedError("This minimal spec supports effects coding only for now.")

        J = int(prep.J)
        p = J - 1
        if p < 1:
            raise ValueError("Need at least 2 items for effects coding (J >= 2).")

        # Build the structured prior covariance for effects coding (Appendix-C style)
        beta_struct_prior_cov = _effects_prior_cov(J=J, v=v_prior)  # (p, p)

        # Inverse-Wishart hyperparameters for D
        iw_df = p + int(df_add)
        if iw_df <= p - 1:
            raise ValueError("Inverse-Wishart df must exceed p-1. Use df_add >= 0 (typ. 5).")

        # Use the structured prior covariance as a sensible scale for IW
        # (You can scale this if you need a specific prior mean for D.)
        iw_scale = beta_struct_prior_cov.copy()

        # Effects expander: append the negative sum column to enforce sum-to-zero
        def effects_expand(beta_tilde: np.ndarray) -> np.ndarray:
            # beta_tilde: (p,) for the first J-1 levels
            b_last = -np.sum(beta_tilde, axis=-1, keepdims=False)
            return np.concatenate([beta_tilde, np.array([b_last], dtype=beta_tilde.dtype)], axis=-1)

        # Basic sanity checks on PreparedMaxDiff (types/sizes)
        _quick_prepared_checks(prep, J=J)

        return HBModelSpec(
            # Universe & coding
            J=J,
            p=p,
            item_ids=list(prep.item_ids),
            coding="effects",

            # Event structure
            N=int(prep.N),
            E=int(prep.E),
            respondent_idx=_as_dtype(prep.respondent_idx, np.int32),
            stage=_as_dtype(prep.stage, np.uint8),
            set_ptr=_as_dtype(prep.set_ptr, np.int32),
            set_items=_as_dtype(prep.set_items, np.int32),
            chosen_item=_as_dtype(prep.chosen_item, np.int32),
            weight=_as_dtype(prep.weight, np.float32),

            # Priors
            v_prior=float(v_prior),
            df_add=int(df_add),
            iw_df=int(iw_df),
            beta_struct_prior_cov=beta_struct_prior_cov,
            iw_scale=iw_scale,

            # Alpha prior
            alpha_prior_tau=float(alpha_prior_tau),

            # Utilities
            effects_expand=effects_expand,

            # MCMC knobs
            burn_in=int(burn_in),
            iters=int(iters),
            thin=int(thin),
            mh_step_init=float(mh_step_init),
            target_accept=float(target_accept),
        )


# ---- 3) Small helpers --------------------------------------------------------

def _effects_prior_cov(J: int, v: float) -> np.ndarray:
    """
    Build the (J-1)x(J-1) covariance matrix corresponding to effects coding
    for a single 'attribute' with J levels, following Sawtooth CBC/HB structure:
      diag = (J-1)*v / J
      off  = -v / J
    """
    p = J - 1
    diag_val = (J - 1) * v / J
    off_val = -v / J
    cov = np.full((p, p), off_val, dtype=float)
    np.fill_diagonal(cov, diag_val)
    return cov


def _as_dtype(arr: np.ndarray, dtype) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype != dtype:
        a = a.astype(dtype, copy=False)
    return a


def _quick_prepared_checks(prep, J: int) -> None:
    # Minimal, fast checks to catch accidental dtype/shape drift.
    if prep.set_ptr.ndim != 1 or prep.set_items.ndim != 1:
        raise ValueError("set_ptr must be 1D (E+1,), set_items must be 1D (sum set sizes).")
    if prep.set_ptr[0] != 0 or prep.set_ptr[-1] != len(prep.set_items):
        raise ValueError("CSR invariants violated: set_ptr[0]=0 and set_ptr[-1]=len(set_items).")
    if np.any(prep.set_items < 0) or np.any(prep.set_items >= J):
        raise ValueError("set_items contains out-of-range item indices.")
    if np.any(prep.chosen_item < 0) or np.any(prep.chosen_item >= J):
        raise ValueError("chosen_item contains out-of-range item indices.")
    if len(prep.respondent_idx) != len(prep.stage) or len(prep.stage) != len(prep.chosen_item):
        raise ValueError("Event-level arrays must have equal length E.")
