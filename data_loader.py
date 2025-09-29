# data_loader.py
import re
from pathlib import Path
import pandas as pd
import ast


class DataLoader:
    def load_responses(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        print(f"[load_responses] rows={len(df)}, cols={len(df.columns)}")
        return df

    def load_design(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        print(f"[load_design] rows={len(df)}, cols={len(df.columns)}")
        return df

    def combine(self, responses: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
        # required cols
        for c in ["sys_RespNum", "sys_MXDVersion_MXD", "weegvar"]:
            if c not in responses.columns:
                raise KeyError(f"Missing '{c}' in responses.")
        if not {"Version", "Set"}.issubset(design.columns):
            raise KeyError("Design needs 'Version' and 'Set' columns.")

        vcol = "sys_MXDVersion_MXD"  # the ONLY version key; must match design['Version']

        # collect MXD task columns
        b_cols = sorted([c for c in responses.columns if re.fullmatch(r"MXD_\d+_b", c)],
                        key=lambda x: int(re.search(r"\d+", x).group()))
        w_cols = sorted([c for c in responses.columns if re.fullmatch(r"MXD_\d+_w", c)],
                        key=lambda x: int(re.search(r"\d+", x).group()))
        if not b_cols or not w_cols:
            raise KeyError("No MXD_{t}_b / MXD_{t}_w columns found.")

        tasks = sorted(set(int(re.search(r"\d+", c).group()) for c in b_cols)
                       .intersection(int(re.search(r"\d+", c).group()) for c in w_cols))
        b_cols = [f"MXD_{t}_b" for t in tasks]
        w_cols = [f"MXD_{t}_w" for t in tasks]

        base = ["sys_RespNum", vcol, "weegvar"]
        b = responses[base + b_cols].set_index(base)
        w = responses[base + w_cols].set_index(base)
        b.columns = [int(re.search(r"\d+", c).group()) for c in b.columns]
        w.columns = [int(re.search(r"\d+", c).group()) for c in w.columns]

        b_long = b.stack().reset_index()
        b_long.columns = ["sys_RespNum", vcol, "weegvar", "task_id", "best_item"]
        w_long = w.stack().reset_index()
        w_long.columns = ["sys_RespNum", vcol, "weegvar", "task_id", "worst_item"]

        long_df = pd.merge(b_long, w_long, on=["sys_RespNum", vcol, "weegvar", "task_id"], how="inner")

        # numeric coercion
        for c in ["best_item", "worst_item", vcol, "task_id"]:
            long_df[c] = pd.to_numeric(long_df[c], errors="coerce")
        long_df = long_df.dropna(subset=["best_item", "worst_item", vcol, "task_id"]).copy()
        long_df[["best_item", "worst_item", vcol, "task_id"]] = (
            long_df[["best_item", "worst_item", vcol, "task_id"]].astype(int)
        )

        # --- build (Version, Set) -> set_items map to avoid any accidental joins on respondent_id ---
        item_cols = sorted([c for c in design.columns if c.lower().startswith("item")],
                           key=lambda x: int(re.search(r"\d+", x).group()))
        dsg = design.copy()
        dsg["Version"] = pd.to_numeric(dsg["Version"], errors="coerce").astype("Int64")
        dsg["Set"] = pd.to_numeric(dsg["Set"], errors="coerce").astype("Int64")

        def row_items(r):
            vals = [pd.to_numeric(r[c], errors="coerce") for c in item_cols]
            return [int(x) for x in vals if pd.notna(x)]

        items_map = (
            dsg[["Version", "Set"] + item_cols]
            .assign(set_items=lambda x: x.apply(row_items, axis=1))
            .set_index(["Version", "Set"])["set_items"]
            .to_dict()
        )

        long_df["set_items"] = long_df.apply(
            lambda r: items_map.get((int(r[vcol]), int(r["task_id"])), []), axis=1
        )

        out = (
            long_df[["sys_RespNum", vcol, "task_id", "set_items", "best_item", "worst_item", "weegvar"]]
            .rename(columns={"sys_RespNum": "respondent_id", vcol: "version", "weegvar": "weight"})
            .sort_values(["respondent_id", "task_id"])
            .reset_index(drop=True)
        )

        K = len(item_cols)
        print(f"[combine] rows={len(out)}, K={K}")
        return out
    
    def validate(self, responses: pd.DataFrame, design: pd.DataFrame, combined: pd.DataFrame) -> None:
        """
        Quick integrity checks:
        1) best/worst equal raw responses
        2) set_items equal design (Version, Set)
        3) each respondent has correct #tasks for their version
        """
        # --- 1) Compare best/worst with original response columns per task.
        bw_errs = []
        max_t = int(combined["task_id"].max())
        for t in range(1, max_t + 1):
            bcol, wcol = f"MXD_{t}_b", f"MXD_{t}_w"
            if bcol not in responses.columns or wcol not in responses.columns:
                continue
            m = combined[combined["task_id"] == t].merge(
                responses[["sys_RespNum", "sys_MXDVersion_MXD", bcol, wcol]],
                left_on=["respondent_id", "version"],
                right_on=["sys_RespNum", "sys_MXDVersion_MXD"],
                how="left",
            )
            mis = (m["best_item"] != pd.to_numeric(m[bcol], errors="coerce")) | \
                  (m["worst_item"] != pd.to_numeric(m[wcol], errors="coerce"))
            if mis.any():
                bw_errs.append((t, int(mis.sum())))

        # --- 2) set_items vs design using true list equality (robust to string/list types).
        item_cols = sorted(
            [c for c in design.columns if c.lower().startswith("item")],
            key=lambda x: int(re.search(r"\d+", x).group())
        )
        design_ref = (
            design.assign(set_items=design[item_cols].apply(
                lambda r: [int(x) for x in pd.to_numeric(r, errors="coerce") if pd.notna(x)], axis=1))
            .set_index(["Version", "Set"])["set_items"]
            .to_dict()
        )

        def _parse_items(val):
            # Accept list[int], list[str], or string "[..]"; fall back to extracting digits.
            if isinstance(val, list):
                return [int(x) for x in val]
            if pd.isna(val):
                return []
            if isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        return [int(x) for x in parsed]
                except Exception:
                    pass
                # fallback: extract integers from any string
                return [int(s) for s in re.findall(r"-?\d+", val)]
            return []

        set_errs = []
        for _, r in combined.iterrows():
            expected = design_ref.get((r["version"], r["task_id"]))
            actual = _parse_items(r["set_items"])
            if expected is None or actual != expected:
                set_errs.append((r["respondent_id"], r["version"], r["task_id"]))

        # --- 3) per-respondent task counts vs design.
        expected_tasks = design.groupby("Version")["Set"].nunique().to_dict()
        counts = combined.groupby(["respondent_id", "version"])["task_id"].nunique()
        bad_counts = counts[counts != counts.index.get_level_values("version").map(expected_tasks)]

        print("Best/Worst mismatches per task:", bw_errs or "None")
        print("Set_items mismatches:", set_errs or "None")
        print("Row count mismatches:", "None" if bad_counts.empty else bad_counts.to_dict())
