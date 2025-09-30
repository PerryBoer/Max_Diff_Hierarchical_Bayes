"""
MaxDiff Sequential Best–Worst (Sawtooth-style) Preparation
Minimal, modular, pythonic single-file skeleton to transform a validated
compact MaxDiff CSV into a model-ready, alternative-level table and (optionally)
a sparse item-indicator matrix for multinomial logit / HB-MNL estimation.

This file intentionally focuses on clarity:
- Full variable names, no abbreviations.
- No estimation (no MCMC) – just data preparation.
- Straightforward pandas + numpy (+ optional scipy.sparse).

Expected input CSV schema (one row per respondent × task):
- respondent_id : int
- version       : int
- task_id       : int
- set_items     : list-like of item IDs (e.g., "[18,28,25,32,7]")
- best_item     : int (chosen as best)
- worst_item    : int (chosen as worst)
- weight        : float (respondent weight)

Outputs:
1) choice_event_table (two rows per original row: best-stage, worst-stage)
2) alternative_level_table (one row per choice event × alternative)
3) optional sparse indicator matrix X (rows=alternatives, cols=items or items-1)

Run this file directly for a tiny end-to-end demo that prints heads and shapes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import ast
import numpy as np
import pandas as pd

try:
    from scipy import sparse as scipy_sparse
except Exception:  # scipy is optional for the indicator matrix
    scipy_sparse = None


# ========================= Data Schemas ========================= #

@dataclass
class CompactRow:
    respondent_id: int
    version: int
    task_id: int
    set_items: List[int]
    best_item: int
    worst_item: int
    weight: float


@dataclass
class ChoiceEventRow:
    respondent_index: int
    original_respondent_id: int
    version: int
    task_id: int
    stage: str  # "best" or "worst"
    choice_set_item_index_list: List[int]
    chosen_item_index: int
    weight: float


# ========================= Reader ========================= #

def load_maxdiff_compact_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load the validated compact CSV and coerce types.

    The set_items column is parsed into a List[int] even if it is stored as a string.
    """
    df = pd.read_csv(csv_path)

    required_columns = [
        "respondent_id",
        "version",
        "task_id",
        "set_items",
        "best_item",
        "worst_item",
        "weight",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Coerce set_items to List[int]
    def parse_set_items(value) -> List[int]:
        if isinstance(value, list):
            return [int(v) for v in value]
        if isinstance(value, str):
            value = value.strip()
            # Accept formats like "[1,2,3]" or "1,2,3"
            if value.startswith("[") and value.endswith("]"):
                try:
                    parsed = ast.literal_eval(value)
                    return [int(v) for v in parsed]
                except Exception as exc:
                    raise ValueError(f"Could not parse set_items string: {value}") from exc
            # Fallback: comma-separated without brackets
            if "," in value:
                return [int(v.strip()) for v in value.split(",") if v.strip()]
        # If it is numeric or anything unexpected
        raise ValueError(f"set_items must be list-like; got: {value!r}")

    df = df.copy()
    df["set_items"] = df["set_items"].map(parse_set_items)

    # Enforce dtypes
    df["respondent_id"] = df["respondent_id"].astype(int)
    df["version"] = df["version"].astype(int)
    df["task_id"] = df["task_id"].astype(int)
    df["best_item"] = df["best_item"].astype(int)
    df["worst_item"] = df["worst_item"].astype(int)
    df["weight"] = df["weight"].astype(float)

    return df


# ========================= Indexers ========================= #

def build_item_index_map(compact_table: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Build a mapping item_id → 0..J-1 and its inverse.
    Includes items appearing in sets and as best/worst.
    """
    unique_items = set()
    for set_list in compact_table["set_items"]:
        unique_items.update(set_list)
    unique_items.update(compact_table["best_item"].tolist())
    unique_items.update(compact_table["worst_item"].tolist())

    sorted_items = sorted(int(x) for x in unique_items)
    item_to_index = {item_id: i for i, item_id in enumerate(sorted_items)}
    index_to_item = {i: item_id for item_id, i in item_to_index.items()}
    return item_to_index, index_to_item


def build_respondent_index_map(compact_table: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Build a mapping respondent_id → 0..N-1 and its inverse."""
    unique_respondents = sorted(compact_table["respondent_id"].unique().tolist())
    respondent_to_index = {rid: i for i, rid in enumerate(unique_respondents)}
    index_to_respondent = {i: rid for rid, i in respondent_to_index.items()}
    return respondent_to_index, index_to_respondent


# ========================= Sequential Expansion ========================= #

def make_choice_event_rows(
    compact_table: pd.DataFrame,
    item_to_index: Dict[int, int],
    respondent_to_index: Dict[int, int],
) -> pd.DataFrame:
    """Create two sequential choice events per compact row (best-stage, worst-stage).

    Best stage: choice over the full set.
    Worst stage: choice over the set with the best choice removed.

    Returns a DataFrame `choice_event_table` with columns:
        - respondent_index
        - original_respondent_id
        - version
        - task_id
        - stage ("best" or "worst")
        - choice_set_item_index_list (List[int])
        - chosen_item_index (int)
        - weight (float)
    """
    records: List[Dict] = []

    for _, row in compact_table.iterrows():
        original_respondent_id = int(row["respondent_id"])
        respondent_index = respondent_to_index[original_respondent_id]
        version = int(row["version"])  # kept for auditing
        task_id = int(row["task_id"])
        weight = float(row["weight"])

        set_items_list = [int(x) for x in row["set_items"]]
        best_item = int(row["best_item"])
        worst_item = int(row["worst_item"])

        # Map to indices
        choice_set_best = [item_to_index[i] for i in set_items_list]
        chosen_best_index = item_to_index[best_item]

        # Best-stage event
        records.append(
            {
                "respondent_index": respondent_index,
                "original_respondent_id": original_respondent_id,
                "version": version,
                "task_id": task_id,
                "stage": "best",
                "choice_set_item_index_list": choice_set_best,
                "chosen_item_index": chosen_best_index,
                "weight": weight,
            }
        )

        # Worst-stage event (remove best from the set)
        reduced_set_items = [i for i in set_items_list if i != best_item]
        choice_set_worst = [item_to_index[i] for i in reduced_set_items]
        chosen_worst_index = item_to_index[worst_item]

        records.append(
            {
                "respondent_index": respondent_index,
                "original_respondent_id": original_respondent_id,
                "version": version,
                "task_id": task_id,
                "stage": "worst",
                "choice_set_item_index_list": choice_set_worst,
                "chosen_item_index": chosen_worst_index,
                "weight": weight,
            }
        )

    choice_event_table = pd.DataFrame.from_records(records)
    return choice_event_table


# ========================= Alternative-Level Table ========================= #

def build_alternative_level_table(choice_event_table: pd.DataFrame) -> pd.DataFrame:
    """Explode choice events into alternative-level rows.

    Output columns:
        - respondent_index
        - original_respondent_id
        - version
        - task_id
        - stage
        - alternative_item_index
        - is_chosen (0/1)
        - weight
    """
    rows: List[Dict] = []

    for _, event in choice_event_table.iterrows():
        respondent_index = int(event["respondent_index"])
        original_respondent_id = int(event["original_respondent_id"])
        version = int(event["version"])
        task_id = int(event["task_id"])
        stage = str(event["stage"])  # "best" or "worst"
        weight = float(event["weight"])  # carry forward

        chosen_item_index = int(event["chosen_item_index"])
        choice_set_item_index_list = list(event["choice_set_item_index_list"])  # List[int]

        for alternative_item_index in choice_set_item_index_list:
            is_chosen = 1 if alternative_item_index == chosen_item_index else 0
            rows.append(
                {
                    "respondent_index": respondent_index,
                    "original_respondent_id": original_respondent_id,
                    "version": version,
                    "task_id": task_id,
                    "stage": stage,
                    "alternative_item_index": int(alternative_item_index),
                    "is_chosen": is_chosen,
                    "weight": weight,
                }
            )

    alternative_level_table = pd.DataFrame.from_records(rows)
    return alternative_level_table


# ========================= Optional: Sparse Indicator Matrix ========================= #

def make_sparse_item_indicator(
    alternative_level_table: pd.DataFrame,
    number_of_items: int,
    reference_item_index: Optional[int] = None,
):
    """Build a (rows × columns) sparse one-hot indicator matrix for items.

    If `reference_item_index` is provided, the corresponding column is dropped
    for identification via reference-item coding.

    Returns (X, column_item_indices) where X is csr_matrix or a dense numpy
    fallback if scipy is unavailable.
    """
    row_indices = np.arange(len(alternative_level_table), dtype=int)
    item_indices = alternative_level_table["alternative_item_index"].to_numpy(dtype=int)

    if reference_item_index is not None:
        # Map original item indices to compacted column indices skipping the reference
        kept_item_indices = [i for i in range(number_of_items) if i != reference_item_index]
        item_to_column = {item_ix: col for col, item_ix in enumerate(kept_item_indices)}
        column_indices = np.array([item_to_column[i] for i in item_indices], dtype=int)
        number_of_columns = number_of_items - 1
        column_item_indices = np.array(kept_item_indices, dtype=int)
    else:
        column_indices = item_indices.copy()
        number_of_columns = number_of_items
        column_item_indices = np.arange(number_of_items, dtype=int)

    data = np.ones(len(row_indices), dtype=float)

    if scipy_sparse is not None:
        X = scipy_sparse.csr_matrix((data, (row_indices, column_indices)),
                                    shape=(len(row_indices), number_of_columns))
    else:
        # Dense fallback with a clear warning via print (no hidden behavior)
        print("[Warning] scipy.sparse not available – building a dense indicator matrix.")
        X = np.zeros((len(row_indices), number_of_columns), dtype=float)
        X[row_indices, column_indices] = 1.0

    return X, column_item_indices


# ========================= Validation Checks ========================= #

def validate_sequential_structure(
    compact_table: pd.DataFrame,
    choice_event_table: pd.DataFrame,
) -> None:
    """Run basic assertions to ensure the sequential expansion is consistent."""
    # 1) Two events per original row
    expected_events = 2 * len(compact_table)
    actual_events = len(choice_event_table)
    if expected_events != actual_events:
        raise AssertionError(f"Expected {expected_events} choice events, found {actual_events}.")

    # 2) Best-stage uses full set; Worst-stage uses set minus best
    grouped = choice_event_table.groupby(["original_respondent_id", "task_id"], sort=False)

    for (orig_rid, task_id), grp in grouped:
        if set(grp["stage"]) != {"best", "worst"}:
            raise AssertionError(
                f"Respondent {orig_rid}, task {task_id}: must have exactly one best and one worst stage."
            )
        best_row = grp.loc[grp["stage"] == "best"].iloc[0]
        worst_row = grp.loc[grp["stage"] == "worst"].iloc[0]

        set_best: List[int] = list(best_row["choice_set_item_index_list"])  # indices
        set_worst: List[int] = list(worst_row["choice_set_item_index_list"])  # indices

        # Worst set must be best set without the chosen best
        chosen_best = int(best_row["chosen_item_index"])
        expected_worst_set = [i for i in set_best if i != chosen_best]
        if sorted(set_worst) != sorted(expected_worst_set):
            raise AssertionError(
                f"Respondent {orig_rid}, task {task_id}: worst-stage set must equal best-stage set without the chosen best."
            )

        # Chosen items must be in their sets
        chosen_worst = int(worst_row["chosen_item_index"]) 
        if chosen_best not in set_best:
            raise AssertionError(
                f"Respondent {orig_rid}, task {task_id}: chosen best not in best-stage set."
            )
        if chosen_worst not in set_worst:
            raise AssertionError(
                f"Respondent {orig_rid}, task {task_id}: chosen worst not in worst-stage set."
            )


# ========================= End-to-end demo ========================= #

def run_demo(csv_path: str | Path,
             reference_item_index: Optional[int] = None) -> None:
    """Small demonstration: load, index, expand, explode to alternatives, and (optionally) build X."""
    print("\n[1/5] Loading compact CSV…")
    compact_table = load_maxdiff_compact_csv(csv_path)
    print(compact_table.head())

    print("\n[2/5] Building index maps…")
    item_to_index, index_to_item = build_item_index_map(compact_table)
    respondent_to_index, index_to_respondent = build_respondent_index_map(compact_table)
    number_of_items = len(item_to_index)
    number_of_respondents = len(respondent_to_index)
    print(f"Items: {number_of_items} | Respondents: {number_of_respondents}")

    print("\n[3/5] Creating sequential choice events (best-stage, worst-stage)…")
    choice_event_table = make_choice_event_rows(compact_table, item_to_index, respondent_to_index)
    print(choice_event_table.head(6))
    print(f"choice_event_table shape: {choice_event_table.shape}")

    print("\n[4/5] Validating sequential structure…")
    validate_sequential_structure(compact_table, choice_event_table)
    print("Validation passed.")

    print("\n[5/5] Building alternative-level table…")
    alternative_level_table = build_alternative_level_table(choice_event_table)
    print(alternative_level_table.head(10))
    print(f"alternative_level_table shape: {alternative_level_table.shape}")

    if reference_item_index is not None or scipy_sparse is not None:
        print("\n[Optional] Building sparse item-indicator matrix X…")
        X, column_item_indices = make_sparse_item_indicator(
            alternative_level_table,
            number_of_items=number_of_items,
            reference_item_index=reference_item_index,
        )
        print(f"Indicator matrix type: {type(X).__name__}")
        print(f"Indicator matrix shape: {getattr(X, 'shape', None)}")
        print(f"Columns represent item indices: {column_item_indices[:10]}{'…' if len(column_item_indices) > 10 else ''}")


if __name__ == "__main__":
    # Example usage: point to your validated compact CSV.
    # You can run:  python maxdiff_prep.py
    # and edit the path below, or pass via sys.argv in your own wrapper.
    demo_csv_path = Path("maxdiff_compact.csv")
    if not demo_csv_path.exists():
        print("[Info] Demo CSV 'maxdiff_compact.csv' not found in current directory.")
        print("       Create it or change 'demo_csv_path' to your actual file path.")
    else:
        # Example: use reference_item_index=0 to drop item 0 column (identification) – optional here
        run_demo(demo_csv_path, reference_item_index=None)
