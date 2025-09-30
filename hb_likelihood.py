# hb_likelihood.py
# -----------------------------------------------------------------------------
# Minimal, fast, and numerically stable likelihood utilities for MaxDiff HB.
# Public API:
#   - expand_effects
#   - loglik_event_from_beta
#   - loglik_respondent
#   - loglik_dataset
#   - rlh_and_pctcert_from_loglik
#
# This module is self-contained (NumPy only) and includes tiny tests at the end.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, Iterable

import numpy as np

__all__ = [
    "expand_effects",
    "loglik_event_from_beta",
    "loglik_respondent",
    "loglik_dataset",
    "rlh_and_pctcert_from_loglik",
]

# =============================================================================
# Spec protocol
# =============================================================================


class HBModelSpec(Protocol):
    """
    Structural contract needed by the likelihood functions.

    Required attributes
    -------------------
    J : int
        Number of items in the universe.
    N : int
        Number of respondents.
    respondent_idx : np.ndarray
        Shape (E,), int32. For each event e, which respondent i it belongs to.
    stage : np.ndarray
        Shape (E,), uint8. 0=Best, 1=Worst.
    set_ptr : np.ndarray
        Shape (E+1,), int32. Pointers into `set_items` (CSR-style).
    set_items : np.ndarray
        Shape (set_ptr[-1],), int32. Concatenated item IDs per event.
    chosen_item : np.ndarray
        Shape (E,), int32. Chosen item ID for each event.
    weight : np.ndarray
        Shape (E,), float32/float64. Event weights (use 1.0 if unweighted).

    Optional
    --------
    effects_expand(beta_tilde: np.ndarray) -> np.ndarray
        If present, will be used to expand effects-coded vectors.
    """

    J: int
    N: int
    respondent_idx: np.ndarray
    stage: np.ndarray
    set_ptr: np.ndarray
    set_items: np.ndarray
    chosen_item: np.ndarray
    weight: np.ndarray

    # Optional method:
    # def effects_expand(self, beta_tilde: np.ndarray) -> np.ndarray: ...


# =============================================================================
# Private helpers
# =============================================================================


def _logsumexp(x: np.ndarray) -> float:
    """
    Numerically stable logsumexp for 1D input.

    Parameters
    ----------
    x : np.ndarray
        1D array.

    Returns
    -------
    float
        log(sum(exp(x))) computed stably.
    """
    if x.ndim != 1:
        raise ValueError(f"_logsumexp expects 1D input, got shape {x.shape}")
    m = np.max(x)
    # handle -inf all-around (degenerate) gracefully
    if not np.isfinite(m):
        return m
    return float(m + np.log(np.sum(np.exp(x - m))))


def _validate_int_array(name: str, arr: np.ndarray, ndim: int):
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr)}")
    if arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must have integer dtype, got {arr.dtype}")


def _validate_float_array(name: str, arr: np.ndarray, ndim: int):
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr)}")
    if arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {arr.shape}")
    if not (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.number)):
        raise TypeError(f"{name} must have float dtype, got {arr.dtype}")


# =============================================================================
# Public API
# =============================================================================


def expand_effects(beta_tilde: np.ndarray, J: int) -> np.ndarray:
    """
    Expand effects-coded β̃∈R^{J−1} to full-length β∈R^{J} with sum-to-zero.

    Supports (p,) -> (J,) and (n,p) -> (n,J), where p == J-1.

    Parameters
    ----------
    beta_tilde : np.ndarray
        Effects-coded vector(s). Shape (p,) or (n, p) with p=J-1.
    J : int
        Number of items in the universe.

    Returns
    -------
    np.ndarray
        Expanded β with shape (J,) or (n, J), satisfying β.sum(axis=-1)=0.

    Raises
    ------
    ValueError
        On shape mismatch or invalid J.
    """
    if not isinstance(J, int) or J < 2:
        raise ValueError(f"J must be an integer >= 2, got {J}")

    if not isinstance(beta_tilde, np.ndarray):
        raise TypeError("beta_tilde must be a numpy.ndarray")

    if beta_tilde.ndim == 1:
        p = beta_tilde.shape[0]
        if p != J - 1:
            raise ValueError(f"beta_tilde length {p} != J-1 = {J-1}")
        last = -np.sum(beta_tilde, dtype=float)
        out = np.empty(J, dtype=float)
        out[:-1] = beta_tilde.astype(float, copy=False)
        out[-1] = last
        return out

    if beta_tilde.ndim == 2:
        n, p = beta_tilde.shape
        if p != J - 1:
            raise ValueError(f"beta_tilde.shape[1] {p} != J-1 = {J-1}")
        sums = -np.sum(beta_tilde, axis=1, dtype=float)
        out = np.empty((n, J), dtype=float)
        out[:, :-1] = beta_tilde.astype(float, copy=False)
        out[:, -1] = sums
        return out

    raise ValueError(f"beta_tilde must be 1D or 2D, got shape {beta_tilde.shape}")


def loglik_event_from_beta(
    beta_full: np.ndarray,
    set_items: np.ndarray,
    chosen_item: int,
    stage: int,
) -> float:
    """
    Compute a single event's log-probability given full-length utilities β (len J).

    Best event (stage=0):      log p = u_b - logsumexp(u_set).
    Worst event (stage=1): treat as Best on -u: log p = (-u_w) - logsumexp(-u_set).

    Parameters
    ----------
    beta_full : np.ndarray
        Shape (J,). Full utilities for all items.
    set_items : np.ndarray
        Shape (k,), int. Item IDs in the event's offered set.
    chosen_item : int
        The chosen item ID for this event.
    stage : int
        0 for Best, 1 for Worst.

    Returns
    -------
    float
        The event's log-likelihood contribution (log probability).

    Raises
    ------
    ValueError
        If shapes/dtypes invalid, chosen_item not in set, or stage invalid.
    """
    if not isinstance(beta_full, np.ndarray) or beta_full.ndim != 1:
        raise ValueError("beta_full must be a 1D numpy array of length J")
    J = beta_full.shape[0]

    _validate_int_array("set_items", set_items, ndim=1)

    if not (isinstance(chosen_item, (int, np.integer)) and 0 <= chosen_item < J):
        raise ValueError(f"chosen_item must be int in [0, {J-1}], got {chosen_item}")

    if chosen_item not in set_items:
        raise ValueError(
            f"chosen_item={chosen_item} not in set_items={set_items.tolist()}"
        )

    if stage not in (0, 1):
        raise ValueError(f"stage must be 0 (Best) or 1 (Worst), got {stage}")

    u_set = beta_full[set_items]

    if stage == 0:
        # Best
        u_chosen = beta_full[chosen_item]
        denom = _logsumexp(u_set)
        return float(u_chosen - denom)
    else:
        # Worst -> Best on -u
        u_set_neg = -u_set
        u_chosen_neg = -beta_full[chosen_item]
        denom = _logsumexp(u_set_neg)
        return float(u_chosen_neg - denom)


def loglik_respondent(beta_tilde: np.ndarray, spec: HBModelSpec, i: int) -> float:
    """
    Log-likelihood for respondent i by summing weighted event log-probs.

    Parameters
    ----------
    beta_tilde : np.ndarray
        Effects-coded respondent vector of shape (J-1,).
    spec : HBModelSpec
        Model spec providing event structure and (optionally) effects_expand.
    i : int
        Respondent index in [0..N-1].

    Returns
    -------
    float
        Scalar log-likelihood for respondent i.
    """
    if not (isinstance(i, (int, np.integer)) and 0 <= i < spec.N):
        raise ValueError(f"respondent index i must be in [0, {spec.N-1}], got {i}")

    # expand β̃ -> β (prefer spec.effects_expand if provided)
    if hasattr(spec, "effects_expand") and callable(getattr(spec, "effects_expand")):
        beta_full = spec.effects_expand(beta_tilde)  # type: ignore[attr-defined]
    else:
        beta_full = expand_effects(beta_tilde, spec.J)

    # Validate critical arrays once
    _validate_int_array("respondent_idx", spec.respondent_idx, 1)
    _validate_int_array("stage", spec.stage, 1)
    _validate_int_array("set_ptr", spec.set_ptr, 1)
    _validate_int_array("set_items", spec.set_items, 1)
    _validate_int_array("chosen_item", spec.chosen_item, 1)
    _validate_float_array("weight", spec.weight, 1)

    E = spec.respondent_idx.shape[0]
    if not (spec.stage.shape[0] == E and spec.set_ptr.shape[0] == E + 1):
        raise ValueError("Inconsistent event array shapes in spec.")

    mask = (spec.respondent_idx == int(i))
    if not np.any(mask):
        return 0.0

    # Iterate over selected events (loop is fine; inner math is stable)
    total = 0.0
    # indices of events for respondent i
    ev_idx = np.nonzero(mask)[0]
    for e in ev_idx:
        lo, hi = int(spec.set_ptr[e]), int(spec.set_ptr[e + 1])
        S = spec.set_items[lo:hi]
        chosen = int(spec.chosen_item[e])
        stg = int(spec.stage[e])
        w = float(spec.weight[e])
        ll = loglik_event_from_beta(beta_full, S, chosen, stg)
        total += w * ll
    return float(total)


def loglik_dataset(
    beta_tilde_all: np.ndarray,
    spec: HBModelSpec,
    respondents: Optional[np.ndarray] = None,
) -> float:
    """
    Sum respondent log-likelihoods across all (or a subset of) respondents.

    Parameters
    ----------
    beta_tilde_all : np.ndarray
        Shape (N, J-1). Row i contains β̃ for respondent i.
    spec : HBModelSpec
        Model spec with event structure.
    respondents : np.ndarray | None, optional
        1D array of respondent indices to include. If None, include all 0..N-1.

    Returns
    -------
    float
        Total log-likelihood for the selected respondents.

    Raises
    ------
    ValueError
        On shape mismatch or invalid respondent indices.
    """
    if not isinstance(beta_tilde_all, np.ndarray) or beta_tilde_all.ndim != 2:
        raise ValueError("beta_tilde_all must be a 2D numpy array (N, J-1)")
    if beta_tilde_all.shape != (spec.N, spec.J - 1):
        raise ValueError(
            f"beta_tilde_all shape {beta_tilde_all.shape} != ({spec.N}, {spec.J-1})"
        )

    if respondents is None:
        respondents = np.arange(spec.N, dtype=int)
    else:
        respondents = np.asarray(respondents, dtype=int)
        if respondents.ndim != 1:
            raise ValueError("respondents must be a 1D array of indices")
        if np.any((respondents < 0) | (respondents >= spec.N)):
            bad = respondents[(respondents < 0) | (respondents >= spec.N)]
            raise ValueError(f"respondents contains invalid indices: {bad}")

    total = 0.0
    for i in respondents:
        total += loglik_respondent(beta_tilde_all[i], spec, int(i))
    return float(total)


def rlh_and_pctcert_from_loglik(
    loglik: float,
    spec: HBModelSpec,
    respondents: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Weighted RLH and Percent Certainty.

    RLH = exp( loglik / M_eff ), with M_eff = sum of weights over included events.
    Chance log-likelihood = sum_e w_e * ( -log(|S_e|) ) over included events.
    Pct. Cert. = 1 − (loglik / loglik_chance).
    """
    _validate_int_array("respondent_idx", spec.respondent_idx, 1)
    _validate_int_array("set_ptr", spec.set_ptr, 1)
    _validate_int_array("set_items", spec.set_items, 1)
    _validate_float_array("weight", spec.weight, 1)

    E = spec.respondent_idx.shape[0]
    if spec.set_ptr.shape[0] != E + 1:
        raise ValueError("spec.set_ptr must have length E+1")
    if spec.weight.shape[0] != E:
        raise ValueError("spec.weight must have length E")

    if respondents is None:
        mask = np.ones(E, dtype=bool)
    else:
        respondents = np.asarray(respondents, dtype=int)
        if respondents.ndim != 1:
            raise ValueError("respondents must be a 1D array of indices")
        ok = np.zeros(spec.N, dtype=bool)
        ok[respondents] = True
        mask = ok[spec.respondent_idx]

    if not np.any(mask):
        raise ValueError("No events selected for RLH/PctCert computation.")

    w = spec.weight.astype(float, copy=False)
    if np.any(w[mask] < 0):
        raise ValueError("Weights must be nonnegative.")

    M_eff = float(np.sum(w[mask]))  # effective number of (weighted) events
    if M_eff == 0.0:
        raise ValueError("Sum of selected weights is zero; RLH undefined.")

    chance_loglik = 0.0
    ev_idx = np.nonzero(mask)[0]
    for e in ev_idx:
        lo, hi = int(spec.set_ptr[e]), int(spec.set_ptr[e + 1])
        k = hi - lo
        if k <= 0:
            raise ValueError(f"Event {e} has empty set.")
        chance_loglik += w[e] * (-np.log(float(k)))

    rlh = float(np.exp(loglik / M_eff))
    pct_cert = float(1.0 - (loglik / chance_loglik))
    return rlh, pct_cert


# =============================================================================
# Tiny synthetic fixtures + tests
# =============================================================================


@dataclass
class _TinySpec:
    """Lightweight spec used only for internal tests."""
    J: int
    N: int
    respondent_idx: np.ndarray
    stage: np.ndarray
    set_ptr: np.ndarray
    set_items: np.ndarray
    chosen_item: np.ndarray
    weight: np.ndarray

    # expose optional method to ensure compatibility
    def effects_expand(self, beta_tilde: np.ndarray) -> np.ndarray:
        return expand_effects(beta_tilde, self.J)


def _fixture_A() -> Tuple[_TinySpec, np.ndarray]:
    """
    Balanced tiny case: J=5, one respondent, two events:
      - Event 0: Best, set size 5
      - Event 1: Worst, set size 4 (previous Best removed)
    β̃ = 0 => chance.
    """
    J = 5
    N = 1
    # two events
    respondent_idx = np.array([0, 0], dtype=np.int32)
    stage = np.array([0, 1], dtype=np.uint8)  # Best, Worst
    # Event 0 set: [0,1,2,3,4], choose 0 as Best
    # Event 1 set: [1,2,3,4], choose 1 as Worst (arbitrary but in set)
    set_items = np.array([0, 1, 2, 3, 4, 1, 2, 3, 4], dtype=np.int32)
    set_ptr = np.array([0, 5, 9], dtype=np.int32)
    chosen_item = np.array([0, 1], dtype=np.int32)
    weight = np.array([1.0, 1.0], dtype=np.float64)

    spec = _TinySpec(
        J=J,
        N=N,
        respondent_idx=respondent_idx,
        stage=stage,
        set_ptr=set_ptr,
        set_items=set_items,
        chosen_item=chosen_item,
        weight=weight,
    )
    beta_tilde = np.zeros(J - 1, dtype=float)
    return spec, beta_tilde


def _fixture_B() -> Tuple[_TinySpec, np.ndarray]:
    """
    Deterministic utility spike: one item has +5 utility in Best set.
    Worst event uses same set; if the +5 item is (wrongly) chosen as worst,
    log-lik should be very negative. Monotonicity checks are performed.
    """
    J = 5
    N = 1
    respondent_idx = np.array([0, 0], dtype=np.int32)
    stage = np.array([0, 1], dtype=np.uint8)
    # Both events share the same set [0,1,2,3,4]
    set_items = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int32)
    set_ptr = np.array([0, 5, 10], dtype=np.int32)
    # Best chooses the spiked item 2; Worst chooses item 3 (not the spiked one)
    chosen_item = np.array([2, 3], dtype=np.int32)
    weight = np.array([1.0, 1.0], dtype=np.float64)

    spec = _TinySpec(
        J=J,
        N=N,
        respondent_idx=respondent_idx,
        stage=stage,
        set_ptr=set_ptr,
        set_items=set_items,
        chosen_item=chosen_item,
        weight=weight,
    )
    # build beta_full with spike at item 2 of +5, then convert to β̃
    beta_full = np.zeros(J, dtype=float)
    beta_full[2] = 5.0
    # Convert to effects-coded β̃ by dropping last and ensuring sum to zero:
    # Here: β̃ = β[0:J-1], implying β[J-1] is the omitted effect.
    # For testing monotonicity we can back-calc β̃ as beta_full[:-1],
    # but ensure sum-to-zero by subtracting mean so that β[J-1] = -sum(β̃).
    beta_centered = beta_full - np.mean(beta_full)
    beta_tilde = beta_centered[:-1].copy()
    return spec, beta_tilde


def _fixture_C_weights() -> Tuple[_TinySpec, np.ndarray]:
    """
    Weights: Two identical Best events with weight=2.0 each.
    The dataset log-lik equals 2x the single-event log-lik summed twice -> 4x single-event.
    We'll verify proportionality (twice one event equals one event with weight 2).
    """
    J = 5
    N = 1
    respondent_idx = np.array([0, 0], dtype=np.int32)
    stage = np.array([0, 0], dtype=np.uint8)  # both Best
    # Both events same set [0,1,2,3,4], chosen 0
    set_items = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int32)
    set_ptr = np.array([0, 5, 10], dtype=np.int32)
    chosen_item = np.array([0, 0], dtype=np.int32)
    weight = np.array([2.0, 2.0], dtype=np.float64)

    spec = _TinySpec(
        J=J,
        N=N,
        respondent_idx=respondent_idx,
        stage=stage,
        set_ptr=set_ptr,
        set_items=set_items,
        chosen_item=chosen_item,
        weight=weight,
    )
    beta_tilde = np.zeros(J - 1, dtype=float)  # chance
    return spec, beta_tilde


def _tests():
    # --------------------------
    # Fixture A: chance behavior
    # --------------------------
    specA, beta_tildeA = _fixture_A()
    ll_i = loglik_respondent(beta_tildeA, specA, 0)
    # Expected: -log(5) + -log(4)
    expected = -np.log(5.0) - np.log(4.0)
    assert abs(ll_i - expected) < 1e-10, f"A: ll {ll_i} != {expected}"

    betaA_all = beta_tildeA[None, :]
    ll_ds = loglik_dataset(betaA_all, specA)
    assert abs(ll_ds - expected) < 1e-10, "A: dataset loglik mismatch"

    rlh, pct = rlh_and_pctcert_from_loglik(ll_ds, specA)
    # RLH = sqrt(1/20)
    rlh_expected = np.sqrt(1.0 / 20.0)
    assert abs(rlh - rlh_expected) < 1e-12, f"A: RLH {rlh} != {rlh_expected}"
    # Percent Certainty ~ 0
    assert abs(pct - 0.0) < 1e-12, f"A: pct_cert {pct} != 0"

    # -------------------------------------
    # Fixture B: deterministic utility spike
    # -------------------------------------
    specB, beta_tildeB = _fixture_B()
    betaB_all = beta_tildeB[None, :]

    # Event 0 (Best): chosen item has +5 spike -> log p ~ 0 (very likely)
    # We compute event 0 directly
    beta_full_B = expand_effects(beta_tildeB, specB.J)
    S0 = specB.set_items[specB.set_ptr[0] : specB.set_ptr[1]]
    e0_ll = loglik_event_from_beta(beta_full_B, S0, chosen_item=specB.chosen_item[0], stage=0)
    assert e0_ll > -1e-3, f"B: Best event with spike should be near 0, got {e0_ll}"

    # Event 1 (Worst): chosen item is NOT the +5 item. Since Worst uses -u,
    # picking a non-spike as worst should be relatively plausible (not extremely negative).
    S1 = specB.set_items[specB.set_ptr[1] : specB.set_ptr[2]]
    e1_ll = loglik_event_from_beta(beta_full_B, S1, chosen_item=specB.chosen_item[1], stage=1)
    assert np.isfinite(e1_ll), "B: Worst event ll should be finite"

    # Monotonicity checks:
    # For Best: increasing chosen utility increases log-lik.
    eps = 1e-3
    chosen = specB.chosen_item[0]
    beta_plus = beta_full_B.copy()
    beta_minus = beta_full_B.copy()
    beta_plus[chosen] += eps
    beta_minus[chosen] -= eps
    e0_ll_plus = loglik_event_from_beta(beta_plus, S0, chosen, stage=0)
    e0_ll_minus = loglik_event_from_beta(beta_minus, S0, chosen, stage=0)
    assert e0_ll_plus > e0_ll > e0_ll_minus, "B: Best monotonicity failed"

    # For Worst: decreasing chosen (worst) utility increases log-lik (since we use -u).
    chosen_w = specB.chosen_item[1]
    beta_plus = beta_full_B.copy()
    beta_minus = beta_full_B.copy()
    beta_plus[chosen_w] += eps
    beta_minus[chosen_w] -= eps
    e1_ll_plus = loglik_event_from_beta(beta_plus, S1, chosen_w, stage=1)
    e1_ll_minus = loglik_event_from_beta(beta_minus, S1, chosen_w, stage=1)
    assert e1_ll_minus > e1_ll > e1_ll_plus, "B: Worst monotonicity failed"

    # ---------------------------
    # Fixture C: weight behavior
    # ---------------------------
    specC, beta_tildeC = _fixture_C_weights()
    betaC_all = beta_tildeC[None, :]

    # Single event (unweighted) log-lik for reference:
    beta_full_C = expand_effects(beta_tildeC, specC.J)
    S_ev = specC.set_items[0:5]
    e_ll = loglik_event_from_beta(beta_full_C, S_ev, chosen_item=0, stage=0)

    # Dataset has two identical events with weight 2 each -> total = 2*(2*e_ll) = 4*e_ll
    ll_weighted = loglik_dataset(betaC_all, specC)
    assert abs(ll_weighted - 4.0 * e_ll) < 1e-12, "C: weights proportionality failed"

    # --------------------------------
    # Fixture D: expand_effects shapes
    # --------------------------------
    J = 5
    p = J - 1
    # 1D
    bt = np.array([1.0, -2.0, 0.5, 0.5], dtype=float)
    bf = expand_effects(bt, J)
    assert bf.shape == (J,)
    assert abs(np.sum(bf)) < 1e-12, "D: 1D sum-to-zero failed"

    # 2D
    bt2 = np.vstack([bt, 2 * bt])
    bf2 = expand_effects(bt2, J)
    assert bf2.shape == (2, J)
    assert np.allclose(np.sum(bf2, axis=1), 0.0), "D: 2D sum-to-zero failed"

    # Shape errors
    try:
        expand_effects(np.ones(J, dtype=float), J)
        raise AssertionError("D: expected ValueError for wrong p")
    except ValueError:
        pass

    print("All tests passed.")


if __name__ == "__main__":
    _tests()
