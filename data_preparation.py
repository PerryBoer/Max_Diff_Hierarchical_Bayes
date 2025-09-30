"""
DataPreparation → PreparedMaxDiff (refactored, readable, first principles)

Goal
----
Provide a small, easy-to-read module that converts a validated compact MaxDiff
CSV (one row per respondent×task) into a single immutable PreparedMaxDiff bundle
ready for HB-MNL (Sawtooth-style sequential Best→Worst).

Design choices
--------------
- Clear names and step-by-step comments.
- Minimal public surface:
    * DataPreparation.build_prepared_maxdiff(csv_path, use_holdouts=None)
    * DataPreparation.summary_prepared_maxdiff(prep)
    * DataPreparation.validate(prep)  # Optional, lightweight checks
- Exact dtypes and shapes as required (CSR ragged sets):
    * set_ptr[int32], set_items[int32]
    * respondent_idx[int32], stage[uint8], chosen_item[int32], weight[float32]
- Stable ID orders for items and respondents.
- No estimation, no coding (effects/dummy) here.

Input CSV must contain columns:
    respondent_id:int, version:int, task_id:int,
    set_items:list-like (e.g. "[12,28,5,30,33]"),
    best_item:int, worst_item:int, weight:float
Optional: is_holdout:bool

Outputs in PreparedMaxDiff:
    Universe: item_ids, item_id_to_col, J, respondent_ids, N
    Events: E, respondent_idx, stage, set_ptr, set_items, chosen_item, weight
    Optional: task_id, version, is_holdout
    QA: alt_level_table (long format; not used by sampler)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Tuple
import ast

import numpy as np
import pandas as pd

# ========================= Public dataclass ========================= #

@dataclass(frozen=True)
class PreparedMaxDiff:
    # Universe & IDs
    item_ids: List[int]
    item_id_to_col: Dict[int, int]
    J: int
    respondent_ids: List[Hashable]
    N: int

    # Event structure (sequential Best→Worst)
    E: int
    respondent_idx: np.ndarray  # int32, shape (E,)
    stage: np.ndarray           # uint8, shape (E,), 0=Best, 1=Worst
    set_ptr: np.ndarray         # int32, shape (E+1,)
    set_items: np.ndarray       # int32, shape (set_ptr[-1],)
    chosen_item: np.ndarray     # int32, shape (E,)
    weight: np.ndarray          # float32, shape (E,)

    # Optional (QA/holdouts)
    task_id: Optional[np.ndarray] = None  # int32, shape (E,)
    version: Optional[np.ndarray] = None  # int16, shape (E,)
    is_holdout: Optional[np.ndarray] = None  # bool, shape (E,)

    # For QA/reporting only (not consumed by the sampler)
    alt_level_table: Optional[pd.DataFrame] = field(default=None, compare=False, repr=False)


# ========================= DataPreparation ========================= #

class DataPreparation:
    """Builds the immutable PreparedMaxDiff bundle from a compact CSV.

    Public methods
    --------------
    build_prepared_maxdiff(csv_path, use_holdouts=None) -> PreparedMaxDiff
        Main entry point: loads CSV, expands sequential events, emits CSR arrays.

    summary_prepared_maxdiff(prep) -> dict
        Tiny summary for logs/QA.

    validate(prep) -> None
        Optional invariants and dtype checks. Raises AssertionError on failure.
    """

    # ---------- Public API ---------- #
    def build_prepared_maxdiff(self, csv_path: str | Path, *, use_holdouts: Optional[bool] = None) -> PreparedMaxDiff:
        """Load, expand, and package the data into PreparedMaxDiff.
        Parameters
        ----------
        csv_path : str | Path
            Path to validated compact CSV.
        use_holdouts : Optional[bool]
            If True and column exists → keep only holdouts.
            If False and column exists → keep only non-holdouts.
            If None → keep all (or if column missing).
        """
        compact = self._load_compact(csv_path)

        # 1) Build stable ID spaces (contiguous indices)
        item_to_col, col_to_item = self._build_item_indices(compact)
        resp_to_row, row_to_resp = self._build_respondent_indices(compact)
        J = len(item_to_col)
        N = len(resp_to_row)

        # 2) Create two events per task: Best (full set), Worst (set minus Best)
        event_table = self._make_event_table(compact, item_to_col, resp_to_row)

        # 3) Optional: filter on holdouts at the event level
        if use_holdouts is not None and "is_holdout" in event_table.columns:
            mask = event_table["is_holdout"].astype(bool).to_numpy()
            event_table = (
                event_table.loc[mask].reset_index(drop=True)
                if use_holdouts
                else event_table.loc[~mask].reset_index(drop=True)
            )

        # 4) Sort deterministically: respondent, task, stage(best then worst)
        event_table = self._sort_events(event_table)

        # 5) Convert events to CSR ragged representation
        respondent_idx, stage, set_ptr, set_items, chosen_item, weight, task_id_opt, version_opt, is_holdout_opt = (
            self._events_to_csr(event_table)
        )

        # 6) Build long alternative-level table for QA/reporting (not used by sampler)
        alt_level_table = self._build_alt_level_table(event_table)

        # 7) Freeze universe orders
        item_ids = [col_to_item[j] for j in range(J)]
        respondent_ids = [row_to_resp[i] for i in range(N)]

        # 8) Pack immutable bundle
        prep = PreparedMaxDiff(
            item_ids=item_ids,
            item_id_to_col=item_to_col,
            J=J,
            respondent_ids=respondent_ids,
            N=N,
            E=int(respondent_idx.shape[0]),
            respondent_idx=respondent_idx,
            stage=stage,
            set_ptr=set_ptr,
            set_items=set_items,
            chosen_item=chosen_item,
            weight=weight,
            task_id=task_id_opt,
            version=version_opt,
            is_holdout=is_holdout_opt,
            alt_level_table=alt_level_table,
        )
        return prep

    @staticmethod
    def summary_prepared_maxdiff(prep: PreparedMaxDiff) -> dict:
        """Return small QA stats for logs or dashboards."""
        E = prep.E
        total_items = int(prep.set_ptr[-1]) if E > 0 else 0
        avg_set_size = (total_items / E) if E else 0.0
        pct_worst = float(np.mean(prep.stage == np.uint8(1))) if E else 0.0
        pct_holdout = (
            float(np.mean(prep.is_holdout)) if (prep.is_holdout is not None and E) else None
        )
        return {
            "N": prep.N,
            "J": prep.J,
            "E": prep.E,
            "avg_set_size": avg_set_size,
            "%worst_events": pct_worst,
            "%holdout_events": pct_holdout,
        }

    # Optional: make validations opt-in and explicit
    @staticmethod
    def validate(prep: PreparedMaxDiff) -> None:
        """Optional invariant checks; raises AssertionError if something is off."""
        # dtypes
        assert prep.respondent_idx.dtype == np.int32
        assert prep.stage.dtype == np.uint8
        assert prep.set_ptr.dtype == np.int32
        assert prep.set_items.dtype == np.int32
        assert prep.chosen_item.dtype == np.int32
        assert prep.weight.dtype == np.float32

        # index ranges
        assert prep.N >= 1 and prep.J >= 1
        if prep.E > 0:
            assert prep.respondent_idx.min() >= 0 and prep.respondent_idx.max() == prep.N - 1
            assert prep.set_items.min() >= 0 and prep.set_items.max() == prep.J - 1

        # CSR integrity
        assert prep.set_ptr[0] == 0
        assert np.all(prep.set_ptr[1:] > prep.set_ptr[:-1])  # strictly increasing
        assert prep.set_ptr[-1] == len(prep.set_items)

        # Reconstruct and check a few events (head/tail and midpoints)
        probe_indices = [0, max(0, prep.E // 2), max(0, prep.E - 1)]
        for e in set(probe_indices):
            start, end = int(prep.set_ptr[e]), int(prep.set_ptr[e + 1])
            ev_items = prep.set_items[start:end]
            assert len(ev_items) >= 2
            assert len(np.unique(ev_items)) == len(ev_items)
            assert prep.chosen_item[e] in ev_items
            assert prep.stage[e] in (np.uint8(0), np.uint8(1))

    # ---------- Internals (small, focused helpers) ---------- #

    # 1) Load CSV and coerce types
    @staticmethod
    def _load_compact(csv_path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        required = [
            "respondent_id",
            "version",
            "task_id",
            "set_items",
            "best_item",
            "worst_item",
            "weight",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        def parse_set_items(value) -> List[int]:
            # Accept list, "[1,2,3]", or "1,2,3"
            if isinstance(value, list):
                return [int(v) for v in value]
            if isinstance(value, str):
                s = value.strip()
                if s.startswith("[") and s.endswith("]"):
                    parsed = ast.literal_eval(s)
                    return [int(v) for v in parsed]
                if "," in s:
                    return [int(v.strip()) for v in s.split(",") if v.strip()]
            raise ValueError(f"set_items must be list-like; got {value!r}")

        df = df.copy()
        df["set_items"] = df["set_items"].map(parse_set_items)

        # Basic dtypes
        df["respondent_id"] = df["respondent_id"].astype("int64")
        df["version"] = df["version"].astype("int64")
        df["task_id"] = df["task_id"].astype("int64")
        df["best_item"] = df["best_item"].astype("int64")
        df["worst_item"] = df["worst_item"].astype("int64")
        df["weight"] = df["weight"].astype("float64")
        if "is_holdout" in df.columns:
            df["is_holdout"] = df["is_holdout"].astype(bool)
        return df

    # 2) Build contiguous item indices
    @staticmethod
    def _build_item_indices(df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
        items = set()
        for s in df["set_items"]:
            items.update(s)
        items.update(df["best_item"].tolist())
        items.update(df["worst_item"].tolist())
        item_ids_sorted = sorted(int(x) for x in items)
        item_to_col = {item_id: j for j, item_id in enumerate(item_ids_sorted)}
        col_to_item = {j: item_id for item_id, j in item_to_col.items()}
        return item_to_col, col_to_item

    # 3) Build contiguous respondent indices
    @staticmethod
    def _build_respondent_indices(df: pd.DataFrame) -> Tuple[Dict[Hashable, int], Dict[int, Hashable]]:
        resp_sorted = sorted(df["respondent_id"].unique().tolist())
        to_row = {rid: i for i, rid in enumerate(resp_sorted)}
        to_id = {i: rid for rid, i in to_row.items()}
        return to_row, to_id

    # 4) Create event table: two rows per task (best, worst)
    @staticmethod
    def _make_event_table(df: pd.DataFrame,item_to_col: Dict[int, int], resp_to_row: Dict[Hashable, int]) -> pd.DataFrame:
        records: List[Dict] = []
        for _, r in df.iterrows():
            rid_orig = int(r["respondent_id"])  # keep for audits if needed
            rid = resp_to_row[rid_orig]
            version = int(r["version"])  # audit/pass-through
            task_id = int(r["task_id"])  # audit/pass-through
            weight = float(r.get("weight", 1.0))
            is_holdout = bool(r.get("is_holdout", False))

            set_items = [int(x) for x in r["set_items"]]
            best_item = int(r["best_item"])
            worst_item = int(r["worst_item"])

            set_cols_best = [item_to_col[i] for i in set_items]
            chosen_best_col = item_to_col[best_item]

            # Event A: Best from full set
            records.append(
                {
                    "respondent_index": rid,
                    "original_respondent_id": rid_orig,
                    "version": version,
                    "task_id": task_id,
                    "stage": "best",
                    "choice_set_item_index_list": set_cols_best,
                    "chosen_item_index": chosen_best_col,
                    "weight": weight,
                    "is_holdout": is_holdout,
                }
            )

            # Event B: Worst from set minus Best
            reduced_items = [i for i in set_items if i != best_item]
            set_cols_worst = [item_to_col[i] for i in reduced_items]
            chosen_worst_col = item_to_col[worst_item]

            records.append(
                {
                    "respondent_index": rid,
                    "original_respondent_id": rid_orig,
                    "version": version,
                    "task_id": task_id,
                    "stage": "worst",
                    "choice_set_item_index_list": set_cols_worst,
                    "chosen_item_index": chosen_worst_col,
                    "weight": weight,
                    "is_holdout": is_holdout,
                }
            )
        return pd.DataFrame.from_records(records)

    # 5) Deterministic sort
    @staticmethod
    def _sort_events(event_table: pd.DataFrame) -> pd.DataFrame:
        stage_cat = pd.Categorical(event_table["stage"], categories=["best", "worst"], ordered=True)
        out = (
            event_table.assign(_stage=stage_cat)
            .sort_values(by=["respondent_index", "task_id", "_stage"], kind="mergesort")
            .drop(columns=["_stage"]).reset_index(drop=True)
        )
        return out

    # 6) Convert events → CSR arrays
    @staticmethod
    def _events_to_csr(event_table: pd.DataFrame):
        E = len(event_table)
        respondent_idx = event_table["respondent_index"].to_numpy(dtype=np.int32)
        stage = np.where(event_table["stage"].to_numpy() == "best", 0, 1).astype(np.uint8)
        chosen_item = event_table["chosen_item_index"].to_numpy(dtype=np.int32)
        weight = event_table["weight"].to_numpy(dtype=np.float32)

        task_id_opt = event_table["task_id"].to_numpy(dtype=np.int32) if "task_id" in event_table.columns else None
        version_opt = event_table["version"].to_numpy(dtype=np.int16) if "version" in event_table.columns else None
        is_holdout_opt = event_table["is_holdout"].to_numpy(dtype=bool) if "is_holdout" in event_table.columns else None

        set_ptr = np.zeros(E + 1, dtype=np.int32)
        set_items_list: List[int] = []
        cursor = 0
        for e in range(E):
            # Ensure uniqueness and (preferably) sorted for reproducibility
            items_unique_sorted = sorted(set(int(x) for x in event_table.iloc[e]["choice_set_item_index_list"]))
            set_items_list.extend(items_unique_sorted)
            cursor += len(items_unique_sorted)
            set_ptr[e + 1] = cursor

            # Check chosen exists; if not, fall back to original order to avoid false negative
            if chosen_item[e] not in items_unique_sorted:
                items_orig = [int(x) for x in event_table.iloc[e]["choice_set_item_index_list"]]
                if chosen_item[e] not in items_orig:
                    raise ValueError(f"Chosen item {chosen_item[e]} not in event {e} set.")
                # overwrite this event region with original list and adjust pointer
                start = set_ptr[e]
                set_ptr[e + 1] = start + len(items_orig)
                set_items_list[start: set_ptr[e + 1]] = items_orig

        set_items = np.asarray(set_items_list, dtype=np.int32)
        return respondent_idx, stage, set_ptr, set_items, chosen_item, weight, task_id_opt, version_opt, is_holdout_opt

    # 7) Build long alternative-level table (QA only)
    @staticmethod
    def _build_alt_level_table(event_table: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict] = []
        for _, ev in event_table.iterrows():
            chosen = int(ev["chosen_item_index"]) 
            for item_col in list(ev["choice_set_item_index_list"]):
                rows.append(
                    {
                        "respondent_index": int(ev["respondent_index"]),
                        "original_respondent_id": int(ev["original_respondent_id"]),
                        "version": int(ev["version"]),
                        "task_id": int(ev["task_id"]),
                        "stage": 0 if ev["stage"] == "best" else 1,
                        "alternative_item_index": int(item_col),
                        "is_chosen": int(item_col == chosen),
                        "weight": float(ev["weight"]),
                        "is_holdout": bool(ev.get("is_holdout", False)),
                    }
                )
        alt = pd.DataFrame.from_records(rows)
        if not alt.empty:
            alt["stage"] = alt["stage"].astype("uint8")
            alt["alternative_item_index"] = alt["alternative_item_index"].astype("int32")
            alt["is_chosen"] = alt["is_chosen"].astype("int8")
            alt["weight"] = alt["weight"].astype("float32")
        return alt
