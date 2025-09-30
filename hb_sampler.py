"""
Sawtooth-style Hierarchical Bayes (HB) sampler for MaxDiff (sequential Best→Worst).

Design goals
------------
- Mirror CBC/HB v5.6 behavior: Gibbs for population (alpha, D), MH for individual utilities (beta_tilde_i).
- Effects-coding space of dimension p = J - 1 (sum-to-zero identification handled by the caller when expanding).
- Minimal external dependencies (NumPy only) and clean, readable variable names (few abbreviations).
- Deterministic seeding and reproducible results.
- Running means by default; optional thinned draw storage for diagnostics.

Required companion module
-------------------------
`hb_likelihood.py` providing:
- expand_effects(beta_tilde: np.ndarray, J: int) -> np.ndarray
- loglik_event_from_beta(beta_full, set_items, chosen_item, stage) -> float
- loglik_respondent(beta_tilde, spec, i) -> float
- loglik_dataset(beta_tilde_all, spec, respondents=None) -> float
- rlh_and_pctcert_from_loglik(loglik, spec, respondents=None) -> tuple[float, float]

Expected `spec: HBModelSpec` fields
-----------------------------------
Data (immutable):
- N, J, p
- respondent_idx, stage, set_ptr, set_items, chosen_item, weight  (used by hb_likelihood via `spec`)

Priors / MCMC knobs:
- alpha_tau: Optional[float]   # large → diffuse. If None or np.isinf, we use diffuse update.
- iw_df: float                 # prior degrees of freedom for D (nu0)
- iw_scale: np.ndarray (p,p)   # prior scale matrix for D (S0), SPD. Can encode effects-structure.

Utilities:
- effects_expand: callable (if needed by callers; sampler itself works in effects space)

This file exposes:
- HBSamplerConfig
- HBResults
- HBSampler
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import numpy as np

# --- Import the validated likelihood API ---
from hb_likelihood import (
    loglik_respondent,
    loglik_dataset,
    rlh_and_pctcert_from_loglik,
)


# =============================
# Dataclasses for config/results
# =============================

@dataclass
class HBSamplerConfig:
    burn_in: int = 10000
    iters: int = 20000
    thin: int = 1  # kept for API symmetry; running means do not require thinning

    mh_step_init: float = 0.10
    target_accept: float = 0.30

    random_seed: Optional[int] = 2025

    save_draws: bool = False
    save_thin: int = 10  # store every k-th *post-burn-in* draw if save_draws=True

    parallel_workers: Optional[int] = None  # reserved for future parallel MH sweeps


@dataclass
class HBResults:
    beta_mean: np.ndarray           # shape (N, p)
    alpha_mean: np.ndarray          # shape (p,)
    D_mean: np.ndarray              # shape (p, p)

    traces: Dict[str, np.ndarray]   # keys: 'loglik', 'rlh', 'pct_cert', 'accept_rate', 'param_rms', 'avg_var'

    # Optional thinned draws (only if save_draws=True)
    beta_draws: Optional[np.ndarray] = None   # shape (S, N, p)
    alpha_draws: Optional[np.ndarray] = None  # shape (S, p)
    D_draws: Optional[np.ndarray] = None      # shape (S, p, p)


# =============================
# Sampler
# =============================

class HBSampler:
    def __init__(self, spec, cfg: HBSamplerConfig):
        self.spec = spec
        self.cfg = cfg

        # Dimensions
        self.N: int = int(spec.N)
        self.p: int = int(spec.p)

        # RNG: reproducible, with substreams if needed later
        if cfg.random_seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(cfg.random_seed)

        # State variables (initialized at diffuse priors)
        self.beta_tilde: np.ndarray = np.zeros((self.N, self.p), dtype=float)
        self.alpha: np.ndarray = np.zeros(self.p, dtype=float)
        self.D: np.ndarray = np.eye(self.p, dtype=float)

        # MH proposal step size (scalar multiplier on Cholesky of D)
        self.step_size: float = float(cfg.mh_step_init)

        # Running accumulators (post-burn-in)
        self._mean_beta_accum: Optional[np.ndarray] = None
        self._mean_alpha_accum: Optional[np.ndarray] = None
        self._mean_D_accum: Optional[np.ndarray] = None
        self._mean_counter: int = 0

        # Optional thinned draw storage
        self._beta_draws: List[np.ndarray] = []
        self._alpha_draws: List[np.ndarray] = []
        self._D_draws: List[np.ndarray] = []

        # Diagnostics traces per iteration
        self._trace_loglik: List[float] = []
        self._trace_rlh: List[float] = []
        self._trace_pct: List[float] = []
        self._trace_accept: List[float] = []
        self._trace_rms: List[float] = []
        self._trace_avgvar: List[float] = []

    # ---------- Public API ----------
    def run(self) -> HBResults:
        total_iterations = int(self.cfg.iters)
        burn_in = int(self.cfg.burn_in)
        save_draws = bool(self.cfg.save_draws)
        save_thin = int(self.cfg.save_thin)

        # Main MCMC loop
        for t in range(1, total_iterations + 1):
            # 1) Draw alpha | betas, D
            self.alpha = self._draw_alpha(self.beta_tilde, self.D)

            # 2) Draw D | betas, alpha
            self.D = self._draw_D(self.beta_tilde, self.alpha)

            # Precompute for this iteration
            chol_D = self._robust_cholesky(self.D)
            D_inv = self._chol_inv(chol_D)  # for quadratic forms in prior

            # 3) MH sweep over respondents
            accepted = 0
            for i in range(self.N):
                was_accepted = self._mh_step_beta(i, chol_D, D_inv)
                accepted += int(was_accepted)

            sweep_accept_rate = accepted / max(1, self.N)

            # 4) Diagnostics (whole-dataset LL, RLH, PctCert, RMS, avg var)
            loglik_val = float(loglik_dataset(self.beta_tilde, self.spec))
            rlh_val, pct_val = rlh_and_pctcert_from_loglik(loglik_val, self.spec)
            param_rms_val = float(np.sqrt(np.mean(self.beta_tilde ** 2)))
            avg_var_val = float(np.var(self.beta_tilde, axis=0, ddof=1).mean())

            self._trace_loglik.append(loglik_val)
            self._trace_rlh.append(rlh_val)
            self._trace_pct.append(pct_val)
            self._trace_accept.append(sweep_accept_rate)
            self._trace_rms.append(param_rms_val)
            self._trace_avgvar.append(avg_var_val)

            # 5) Adapt step size during burn-in only
            if t <= burn_in:
                self.step_size = self._tune_step(self.step_size, sweep_accept_rate)

            # 6) Post-burn-in accumulation and optional thinned storage
            if t > burn_in:
                self._accumulate_running_means()
                if save_draws and ((t - burn_in) % save_thin == 0):
                    self._beta_draws.append(self.beta_tilde.copy())
                    self._alpha_draws.append(self.alpha.copy())
                    self._D_draws.append(self.D.copy())

        # Finalize results
        beta_mean, alpha_mean, D_mean = self._finalize_means()

        traces = {
            "loglik": np.asarray(self._trace_loglik, dtype=float),
            "rlh": np.asarray(self._trace_rlh, dtype=float),
            "pct_cert": np.asarray(self._trace_pct, dtype=float),
            "accept_rate": np.asarray(self._trace_accept, dtype=float),
            "param_rms": np.asarray(self._trace_rms, dtype=float),
            "avg_var": np.asarray(self._trace_avgvar, dtype=float),
        }

        if self.cfg.save_draws and self._beta_draws:
            beta_draws = np.asarray(self._beta_draws)
            alpha_draws = np.asarray(self._alpha_draws)
            D_draws = np.asarray(self._D_draws)
        else:
            beta_draws = None
            alpha_draws = None
            D_draws = None

        return HBResults(
            beta_mean=beta_mean,
            alpha_mean=alpha_mean,
            D_mean=D_mean,
            traces=traces,
            beta_draws=beta_draws,
            alpha_draws=alpha_draws,
            D_draws=D_draws,
        )

    # ---------- Internal: Gibbs blocks ----------
    def _draw_alpha(self, beta_tilde: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Diffuse default: alpha | betas, D ~ N(mean=beta_bar, cov=D/N).
        If spec.alpha_tau is finite, use ridge prior N(0, tau I):
            V = (N D^{-1} + tau^{-1} I)^{-1},  mean = V (D^{-1} sum betas).
        """
        N, p = beta_tilde.shape
        beta_bar = beta_tilde.mean(axis=0)

        tau = getattr(self.spec, "alpha_tau", None)
        if tau is None or (isinstance(tau, float) and not np.isfinite(tau)):
            # Diffuse prior: cov = D / N
            chol = self._robust_cholesky(D / max(1, N))
            draw = beta_bar + chol @ self.rng.standard_normal(p)
            return draw
        else:
            # Finite ridge prior
            D_inv = np.linalg.inv(D)
            precision = N * D_inv + (1.0 / float(tau)) * np.eye(p)
            V = np.linalg.inv(precision)
            m = V @ (D_inv @ beta_tilde.sum(axis=0))
            chol = self._robust_cholesky(V)
            draw = m + chol @ self.rng.standard_normal(p)
            return draw

    def _draw_D(self, beta_tilde: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Inverse-Wishart update: D ~ IW(nu0 + N, S0 + scatter).
        scatter = sum_i (beta_i - alpha)(beta_i - alpha)'.
        """
        S0 = np.asarray(self.spec.iw_scale, dtype=float)
        nu0 = float(self.spec.iw_df)

        residuals = beta_tilde - alpha[None, :]
        scatter = residuals.T @ residuals

        nu = nu0 + beta_tilde.shape[0]
        S = S0 + scatter

        # Sample D by sampling Wishart on the precision and inverting
        # If D ~ IW(nu, S), then D^{-1} ~ W(nu, S^{-1}).
        Sinv = np.linalg.inv(S)
        precision = self._sample_wishart(df=nu, scale=Sinv)

        # Invert with symmetry preservation
        D = np.linalg.inv(precision)
        # Symmetrize to reduce numerical drift
        D = 0.5 * (D + D.T)
        return D

    # ---------- Internal: MH for beta_tilde_i ----------
    def _mh_step_beta(self, i: int, chol_D: np.ndarray, D_inv: np.ndarray) -> bool:
        """One MH step for respondent i in effects space.
        Proposal: beta' = beta + step_size * chol(D) @ z,  z ~ N(0, I).
        Accept with probability min(1, r) where
            log r = (logL' - logL) + (-1/2 * Q' + 1/2 * Q),
        and Q(beta) = (beta - alpha)^T D^{-1} (beta - alpha).
        """
        current_beta = self.beta_tilde[i]

        # Current likelihood and prior quadratic
        current_ll = loglik_respondent(current_beta, self.spec, i)
        diff = current_beta - self.alpha
        current_quad = float(diff @ (D_inv @ diff))

        # Propose
        z = self.rng.standard_normal(self.p)
        proposed_beta = current_beta + self.step_size * (chol_D @ z)

        proposed_ll = loglik_respondent(proposed_beta, self.spec, i)
        diff_p = proposed_beta - self.alpha
        proposed_quad = float(diff_p @ (D_inv @ diff_p))

        log_r = (proposed_ll - current_ll) + (-0.5 * proposed_quad + 0.5 * current_quad)
        accept = (np.log(self.rng.random()) < log_r)

        if accept:
            self.beta_tilde[i] = proposed_beta
            return True
        else:
            return False

    # ---------- Internal: Step-size adaptation ----------
    def _tune_step(self, step_size: float, sweep_accept_rate: float) -> float:
        target = float(self.cfg.target_accept)
        if sweep_accept_rate > target:
            step_size *= 1.10
        else:
            step_size *= 0.90
        # Clamp to safe bounds
        return float(np.clip(step_size, 1e-4, 10.0))

    # ---------- Internal: Running means & finalization ----------
    def _accumulate_running_means(self) -> None:
        if self._mean_beta_accum is None:
            self._mean_beta_accum = np.zeros_like(self.beta_tilde)
            self._mean_alpha_accum = np.zeros_like(self.alpha)
            self._mean_D_accum = np.zeros_like(self.D)
            self._mean_counter = 0

        self._mean_beta_accum += self.beta_tilde
        self._mean_alpha_accum += self.alpha
        self._mean_D_accum += self.D
        self._mean_counter += 1

    def _finalize_means(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._mean_counter == 0:
            # No post-burn-in draws; fall back to last state
            return self.beta_tilde.copy(), self.alpha.copy(), self.D.copy()
        inv = 1.0 / float(self._mean_counter)
        beta_mean = self._mean_beta_accum * inv
        alpha_mean = self._mean_alpha_accum * inv
        D_mean = self._mean_D_accum * inv
        return beta_mean, alpha_mean, D_mean

    # ---------- Linear algebra helpers ----------
    @staticmethod
    def _robust_cholesky(A: np.ndarray) -> np.ndarray:
        """Cholesky with minimal jitter if needed."""
        try:
            return np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            # Add small jitter to the diagonal progressively
            diag = np.diag(A)
            jitter = 1e-8 if diag.size == 0 else max(1e-8, 1e-8 * float(np.max(diag)))
            for k in range(8):
                try:
                    return np.linalg.cholesky(A + (jitter * (10 ** k)) * np.eye(A.shape[0]))
                except np.linalg.LinAlgError:
                    continue
            # As a last resort, symmetrize and try once more
            A_sym = 0.5 * (A + A.T) + 1e-6 * np.eye(A.shape[0])
            return np.linalg.cholesky(A_sym)

    @staticmethod
    def _chol_inv(chol: np.ndarray) -> np.ndarray:
        """Invert SPD matrix from its Cholesky factor."""
        # Solve L L^T X = I  →  First solve L Y = I, then L^T X = Y
        p = chol.shape[0]
        I = np.eye(p)
        # forward solve
        Y = np.linalg.solve(chol, I)
        # backward solve
        X = np.linalg.solve(chol.T, Y)
        return X

    def _sample_wishart(self, df: float, scale: np.ndarray) -> np.ndarray:
        """Bartlett decomposition sampler for Wishart(df, scale), df >= p.
        Returns a p×p SPD matrix.
        """
        p = scale.shape[0]
        if df < p:
            raise ValueError("Wishart degrees of freedom must be >= dimension p.")

        # Cholesky of scale
        L = self._robust_cholesky(scale)

        # Construct lower-triangular A with sqrt(chi2) on diagonal and N(0,1) below
        A = np.zeros((p, p))
        for i in range(p):
            A[i, i] = np.sqrt(self.rng.chisquare(df - i))
            if i + 1 < p:
                A[i + 1 :, i] = self.rng.standard_normal(p - i - 1)

        # Wishart sample: W = L @ A @ A^T @ L^T
        LA = L @ A
        W = LA @ LA.T
        # Symmetrize to control numerical drift
        return 0.5 * (W + W.T)


