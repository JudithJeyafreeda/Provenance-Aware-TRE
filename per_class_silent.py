"""Per-gold-class breakdown of the silent_removal / silent_keep results.
"""
import argparse

import pandas as pd


def as_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes"}


def summarize(df):
    n = len(df)
    ans = df[~df.abstained]
    return pd.Series({
        "n": n,
        "coverage": len(ans) / n if n else float("nan"),
        "n_answered": len(ans),
        "accuracy": (ans.answer == ans.gold_relation).mean() if len(ans) else float("nan"),
        "pred_E1_before_E2": (ans.answer == "EVENT1-BEFORE-EVENT2").mean() if len(ans) else float("nan"),
        "pred_E2_before_E1": (ans.answer == "EVENT2-BEFORE-EVENT1").mean() if len(ans) else float("nan"),
        "pred_overlap": (ans.answer == "EVENT1-OVERLAP-EVENT2").mean() if len(ans) else float("nan"),
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controls", nargs="+", required=True)
    p.add_argument("--control_predictions", nargs="+", required=True)
    p.add_argument("--clean_predictions", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    ctrl = pd.concat([pd.read_csv(f) for f in a.controls], ignore_index=True)
    ctrl["example_id"] = ctrl["note_id"].astype(str)
    ctrl["validity_tier"] = ctrl["validity_tier"].fillna("keep")
    cp = pd.concat([pd.read_csv(f)[["example_id", "answer", "abstained"]]
                    for f in a.control_predictions], ignore_index=True)
    cp["example_id"] = cp["example_id"].astype(str)
    m = ctrl.merge(cp, on="example_id", how="inner")
    m["abstained"] = m["abstained"].map(as_bool)

    clean = pd.read_csv(a.clean_predictions)[["example_id", "answer", "abstained"]]
    clean["example_id"] = clean["example_id"].astype(str)
    clean["abstained"] = clean["abstained"].map(as_bool)
    base = ctrl[["base_note_id", "gold_relation", "variant", "validity_tier"]].copy()
    base["example_id"] = base["base_note_id"].astype(str)
    cm = base.merge(clean, on="example_id", how="inner")

    rows = []
    for cond, df in [("edited", m), ("clean_same_notes", cm)]:
        for keys, g in df.groupby(["variant", "validity_tier", "gold_relation"]):
            rows.append({"condition": cond, "variant": keys[0], "tier": keys[1],
                         "gold_relation": keys[2], **summarize(g).to_dict()})
        for keys, g in df.groupby(["variant", "gold_relation"]):
            rows.append({"condition": cond, "variant": keys[0], "tier": "all",
                         "gold_relation": keys[1], **summarize(g).to_dict()})
    out = pd.DataFrame(rows).sort_values(["variant", "tier", "gold_relation", "condition"])
    out.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.round(3).to_string(index=False))
    print(f"\nmerged {len(m)} control rows and {len(cm)} clean rows; wrote {a.out}")


if __name__ == "__main__":
    main()
