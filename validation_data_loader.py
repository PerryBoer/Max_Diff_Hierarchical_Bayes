import pandas as pd
import re
from pathlib import Path
import ast

# file paths
responses_path = Path("./data/M250512 Budget Thuis Multi utility onderzoek MaxDiff_data_volledig.csv")   # Dataset 1
design_path = Path('./data/M250512_MXD_Design.csv')        # Dataset 2
output_path = Path("maxdiff_compact.csv")

# load
responses = pd.read_csv(responses_path)
design = pd.read_csv(design_path)
compact = pd.read_csv(output_path)

# --- 1. Check best/worst consistency ---
errors_bw = []
for t in range(1, compact["task_id"].max() + 1):
    bcol, wcol = f"MXD_{t}_b", f"MXD_{t}_w"
    if bcol not in responses.columns or wcol not in responses.columns:
        continue
    merged = compact[compact["task_id"] == t].merge(
        responses[["sys_RespNum", "sys_MXDVersion_MXD", bcol, wcol]],
        left_on=["respondent_id", "version"],
        right_on=["sys_RespNum", "sys_MXDVersion_MXD"],
        how="left"
    )
    mism = merged[(merged["best_item"] != merged[bcol]) | (merged["worst_item"] != merged[wcol])]
    if not mism.empty:
        errors_bw.append((t, len(mism)))

print("Best/Worst mismatches per task:", errors_bw or "None")

# --- 2. Check set_items consistency ---
item_cols = sorted([c for c in design.columns if c.lower().startswith("item")],
                   key=lambda x: int(re.search(r"\d+", x).group()))

# build reference dict
design_ref = (
    design.assign(set_items=design[item_cols].apply(lambda r: [int(x) for x in r if pd.notna(x)], axis=1))
    .set_index(["Version", "Set"])["set_items"]
    .to_dict()
)

errors_sets = []
for _, row in compact.iterrows():
    expected = design_ref.get((row["version"], row["task_id"]), None)
    actual = ast.literal_eval(row["set_items"])  # turn "[18,28,...]" into [18,28,...]
    if expected is None or actual != expected:
        errors_sets.append((row["respondent_id"], row["version"], row["task_id"]))

print("Set_items mismatches:", errors_sets or "None")



# --- 3. Row counts per respondent ---
expected_tasks = design.groupby("Version")["Set"].nunique().to_dict()
counts = compact.groupby(["respondent_id", "version"])["task_id"].nunique()
bad_counts = counts[counts != counts.index.get_level_values("version").map(expected_tasks)]
print("Row count mismatches:", bad_counts if not bad_counts.empty else "None")