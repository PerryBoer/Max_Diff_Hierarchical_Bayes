# run_dataloader.py
from pathlib import Path
import pandas as pd
import json
import pickle
from misc_functions import *
import time


if __name__ == "__main__":
    # # file paths
    # responses_path = Path("./data/M250512 Budget Thuis Multi utility onderzoek MaxDiff_data_volledig.csv") # response data
    # design_path = Path('./data/M250512_MXD_Design.csv') # design data
    # output_path = Path("maxdiff_compact.csv") # output file

    # # load data class
    # dl = DataLoader()

    # # load raw data from responses and design files
    # resp = dl.load_responses(responses_path)
    # des = dl.load_design(design_path)

    # # combine to long format
    # compact = dl.combine(resp, des)

    # # validate
    # dl.validate(resp, des, compact)

    # # ensure set_items stays as a Python list literal in the CSV cell
    # compact.to_csv(output_path, index=False)

    # # print check
    # print(f"[save] wrote: {output_path.resolve()}")

    # prepare data
    data_prep = DataPreparation().build_prepared_maxdiff("maxdiff_compact.csv", use_holdouts=None)
    DataPreparation.validate(data_prep)  # optional
    print(DataPreparation.summary_prepared_maxdiff(data_prep))

    # prep hb input
    spec = HBModelSpecBuilder.build_from_prepared(data_prep, coding="effects", v_prior=2.0, df_add=5)

    # print summary
    print(summary_spec(spec))

    # call it after building spec
    quick_likelihood_smoke_tests(spec)

    # time the sampler
    
    start_time = time.time()
    # run sampler
    cfg = HBSamplerConfig(burn_in=1000, iters=1000, mh_step_init=0.10, target_accept=0.30,
                          random_seed=1, save_draws=True, save_thin=20)
    sampler = HBSampler(spec, cfg)
    results = sampler.run()
    print(results.traces['rlh'][-10:].mean(), results.traces['pct_cert'][-10:].mean())

    # stop timing and write elapsed time to metadata
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # --- Save arrays (flatten traces dict to arrays; don't store dicts in npz) ---
    np.savez(
        "hb_results_bundle.npz",
        beta_mean=results.beta_mean,
        alpha_mean=results.alpha_mean,
        D_mean=results.D_mean,
        **{f"trace_{k}": v for k, v in results.traces.items()},
        beta_draws=results.beta_draws,
        alpha_draws=results.alpha_draws,
        D_draws=results.D_draws,
    )

    # --- Save config/metadata (json only) ---
    metadata = {
        "config": vars(cfg),
        "N": spec.N,
        "J": spec.J,
        "p": spec.p,
        "elapsed_time_seconds": elapsed_time,
    }

    with open("hb_results_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # --- Save spec snapshot (NO pickle) ---
    # use respondent IDs from your prepared object so exports have real labels
    save_spec_snapshot_npz(spec, data_prep.respondent_ids, "hb_results_spec_snapshot.npz")

    print("[save] wrote hb_results_bundle.npz, hb_results_metadata.json, hb_results_spec_snapshot.npz")
