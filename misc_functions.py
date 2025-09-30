import numpy as np
from data_loader import DataLoader
from data_preparation import DataPreparation
from hb_sampler import HBSampler, HBSamplerConfig
from hb_model_prep import HBModelSpecBuilder
from hb_likelihood import (expand_effects, 
    loglik_event_from_beta,
    loglik_respondent,
    loglik_dataset,
    rlh_and_pctcert_from_loglik,
)


def summary_spec(spec, n_preview=5) -> dict:
    return {
        "Universe": {
            "J": spec.J,
            "p": spec.p,
            "Items": spec.item_ids[:n_preview],
        },
        "Events": {
            "N": spec.N,
            "E": spec.E,
            "respondent_idx": (spec.respondent_idx.shape, spec.respondent_idx.dtype, spec.respondent_idx[:n_preview].tolist()),
            "stage": (spec.stage.shape, spec.stage.dtype, spec.stage[:n_preview].tolist()),
            "set_ptr": (spec.set_ptr.shape, spec.set_ptr.dtype, spec.set_ptr[:n_preview+1].tolist()),
            "set_items": (spec.set_items.shape, spec.set_items.dtype, spec.set_items[:n_preview].tolist(), "..."),
            "chosen_item": (spec.chosen_item.shape, spec.chosen_item.dtype, spec.chosen_item[:n_preview].tolist()),
            "weight": (spec.weight.shape, spec.weight.dtype, spec.weight[:n_preview].tolist()),
        },
        "Priors": {
            "v_prior": spec.v_prior,
            "df_add": spec.df_add,
            "iw_df": spec.iw_df,
            "beta_struct_prior_cov_shape": spec.beta_struct_prior_cov.shape,
            "iw_scale_shape": spec.iw_scale.shape,
        },
        "MCMC": {
            "burn_in": spec.burn_in,
            "iters": spec.iters,
            "thin": spec.thin,
            "mh_step_init": spec.mh_step_init,
            "target_accept": spec.target_accept,
        },
    }

def quick_likelihood_smoke_tests(spec):
    J, N = spec.J, spec.N
    p = J - 1
    beta0 = np.zeros((N, p), dtype=float)

    ll0   = loglik_dataset(beta0, spec)
    rlh0, pct0 = rlh_and_pctcert_from_loglik(ll0, spec)
    assert abs(pct0) < 1e-9, "Chance PctCert should be ~0"
    print(f"[ok] chance: loglik={ll0:.6f}, RLH={rlh0:.6f}, PctCert≈0")

    # subset
    subs = np.arange(min(N, 50), dtype=int)
    ll_sub = loglik_dataset(beta0, spec, respondents=subs)
    rlh_s, pct_s = rlh_and_pctcert_from_loglik(ll_sub, spec, respondents=subs)
    assert abs(pct_s) < 1e-9
    print(f"[ok] subset chance: loglik={ll_sub:.6f}, RLH={rlh_s:.6f}")

    # per-event spot check on first event
    e = 0
    lo, hi = int(spec.set_ptr[e]), int(spec.set_ptr[e+1])
    S = spec.set_items[lo:hi]
    chosen = int(spec.chosen_item[e])
    stg = int(spec.stage[e])
    beta_full0 = expand_effects(np.zeros(p), J)
    lle = loglik_event_from_beta(beta_full0, S, chosen, stg)
    assert np.isfinite(lle)
    print(f"[ok] event 0 lle (chance) ~ -log(|S|) = {-np.log(len(S)):.6f}: got {lle:.6f}")

# --- add this helper near the top of your file (once) ---
def save_spec_snapshot_npz(spec, respondent_ids, path: str = "hb_results_spec_snapshot.npz") -> None:
    """
    Minimal snapshot of what diagnostics & RLH need.
    No pickling, no closures.
    """
    np.savez(
        path,
        # core sizes
        J=np.array([spec.J], dtype=np.int32),
        N=np.array([spec.N], dtype=np.int32),
        E=np.array([spec.E], dtype=np.int32),

        # labels
        item_ids=np.asarray(spec.item_ids, dtype=np.int32),
        respondent_ids=np.asarray(list(respondent_ids), dtype=object),  # keep original IDs

        # event structure for likelihood / RLH
        respondent_idx=spec.respondent_idx.astype(np.int32, copy=False),
        stage=spec.stage.astype(np.uint8, copy=False),
        set_ptr=spec.set_ptr.astype(np.int32, copy=False),
        set_items=spec.set_items.astype(np.int32, copy=False),
        chosen_item=spec.chosen_item.astype(np.int32, copy=False),
        weight=spec.weight.astype(np.float32, copy=False),
    )