#!/usr/bin/env python3
"""Re-judge existing silent_removal edits with an independent judge model.

  python rejudge_silent_controls.py \
    --controls /synthetic_dataset/silent_controls/silent_removal_controls.csv \
    --out_csv  /synthetic_dataset/silent_controls/silent_removal_controls_judge2.csv \
    --api_base <<api_base>> \
    --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
    --workers 8
Run it on silent_removal_controls_test.csv as well for the BERT test partition.
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pandas as pd
from openai import OpenAI

from make_silent_removal_controls import judge


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controls", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--api_base", required=True)
    p.add_argument("--api_key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model", required=True, help="judge model from a different family than the gate")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    df = pd.read_csv(a.controls)
    client = OpenAI(base_url=a.api_base, api_key=a.api_key)
    clients = {"judge": (client, a.model)}
    args = SimpleNamespace()

    def one(r):
        j = judge(clients, args, str(r["text"]), r.get("event1", "the diagnosis"),
                  r.get("event2", "the treatment initiation"))
        return j["determinable"], j["relation"]

    rows = [r for _, r in df.iterrows()]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(one, rows))
    df["judge2_determinable"] = [d for d, _ in res]
    df["judge2_relation"] = [rel for _, rel in res]
    df["validity_tier_judge1"] = df["validity_tier"]
    df["validity_tier"] = ["strict" if d is False else ("implied_only" if d is True else "judge_error")
                           for d, _ in res]
    df.to_csv(a.out_csv, index=False)
    print(df["validity_tier"].value_counts(dropna=False).to_string())
    print(pd.crosstab(df["validity_tier_judge1"], df["validity_tier"]).to_string())
    print(f"wrote {a.out_csv}")


if __name__ == "__main__":
    main()
