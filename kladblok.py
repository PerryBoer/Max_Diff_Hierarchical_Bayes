# run_dataloader.py
from pathlib import Path
import pandas as pd
from data_loader import DataLoader

if __name__ == "__main__":
    # file paths
    responses_path = Path("./data/M250512 Budget Thuis Multi utility onderzoek MaxDiff_data_volledig.csv") # response data
    design_path = Path('./data/M250512_MXD_Design.csv') # design data
    output_path = Path("maxdiff_compact.csv") # output file

    # load data class
    dl = DataLoader()

    # load raw data from responses and design files
    resp = dl.load_responses(responses_path)
    des = dl.load_design(design_path)

    # combine to long format
    compact = dl.combine(resp, des)

    # validate
    dl.validate(resp, des, compact)

    # ensure set_items stays as a Python list literal in the CSV cell
    compact.to_csv(output_path, index=False)
    print(f"[save] wrote: {output_path.resolve()}")

