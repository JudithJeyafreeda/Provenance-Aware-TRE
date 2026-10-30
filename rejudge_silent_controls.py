#!/usr/bin/env python3
"""Re-judge existing silent_removal edits with an independent judge model.

"""
import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from openai import OpenAI

from make_silent_removal_controls import JUDGE_SYSTEM, RELATIONS, event_block, parse_json


def judge(client, model, text, event1, event2, no_think, max_tokens, retries):
    user = event_block(event1, event2) + "Note:\n" + text  # same input as the original judge
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if no_think else None
    last = "no_answer"
    for _ in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=max_tokens,
                messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
                extra_body=extra,
            )
            out = r.choices[0].message.content or ""
        except Exception as exc:
            last = type(exc).__name__
            continue
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
        obj = parse_json(out)
        if not isinstance(obj, dict):
            last = "invalid_json"
            continue
        det = obj.get("determinable")
        if not isinstance(det, bool):
            last = "no_determinable_field"
            continue
        rel = obj.get("relation") if obj.get("relation") in RELATIONS else None
        return {"determinable": det, "relation": None if det is False else rel,
                "evidence": obj.get("evidence"), "error": None}
    return {"determinable": None, "relation": None, "evidence": None, "error": last}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controls", required=True, help="silent_removal_controls.csv")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--api_base", required=True)
    p.add_argument("--api_key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model", required=True, help="judge model from a different family than the gate")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no_think", type=int, default=1)
    p.add_argument("--max_tokens", type=int, default=400)
    p.add_argument("--retries", type=int, default=3)
    a = p.parse_args()

    df = pd.read_csv(a.controls)
    client = OpenAI(base_url=a.api_base, api_key=a.api_key)

    def one(r):
        return judge(client, a.model, str(r["text"]),
                     r.get("event1", "the diagnosis"), r.get("event2", "the treatment initiation"),
                     a.no_think, a.max_tokens, a.retries)

    rows = [r for _, r in df.iterrows()]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(one, rows))

    df["judge2_model"] = a.model
    df["judge2_determinable"] = [x["determinable"] for x in res]
    df["judge2_relation"] = [x["relation"] for x in res]
    df["judge2_evidence"] = [x["evidence"] for x in res]
    df["judge2_error"] = [x["error"] for x in res]
    df["validity_tier_judge1"] = df["validity_tier"]
    df["validity_tier"] = ["strict" if x["determinable"] is False
                           else "implied_only" if x["determinable"] is True else "judge_error" for x in res]
    df.to_csv(a.out_csv, index=False)

    ok = df[df.validity_tier != "judge_error"]
    agree = int((ok.validity_tier == ok.validity_tier_judge1).sum())
    print(pd.crosstab(df["validity_tier_judge1"], df["validity_tier"], margins=True).to_string())
    print(f"\njudge errors: {int((df.validity_tier == 'judge_error').sum())} of {len(df)}")
    print(f"tier agreement with the original judge: {agree} of {len(ok)} ({agree / max(1, len(ok)):.3f})")
    both = df[(df.validity_tier == "strict") & (df.validity_tier_judge1 == "strict")]
    print(f"strict under both judges: {len(both)}")
    # Cohen's kappa on the two-category tier
    if len(ok):
        po = agree / len(ok)
        p1 = (ok.validity_tier_judge1 == "strict").mean()
        p2 = (ok.validity_tier == "strict").mean()
        pe = p1 * p2 + (1 - p1) * (1 - p2)
        print(f"Cohen's kappa: {(po - pe) / (1 - pe):.3f}" if pe < 1 else "kappa undefined")
    print(f"wrote {a.out_csv}")


if __name__ == "__main__":
    main()
