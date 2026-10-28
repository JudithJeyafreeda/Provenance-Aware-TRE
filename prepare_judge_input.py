#!/usr/bin/env python3
"""
prepare_judge_input.py - build notes.jsonl for Judge.py from the repo CSVs.
"""
import csv
import json
import sys

csv.field_size_limit(sys.maxsize)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--show-headers", nargs="+", metavar="CSV")
    p.add_argument("--notes"); p.add_argument("--notes-id", default="patient_id")
    p.add_argument("--notes-text", default="note_text"); p.add_argument("--notes-gold", default="relation")
    p.add_argument("--removal"); p.add_argument("--removal-id", default="patient_id")
    p.add_argument("--removal-text", default="edited_text"); p.add_argument("--removal-tier", default=None)
    p.add_argument("--keep"); p.add_argument("--keep-id", default="patient_id")
    p.add_argument("--keep-text", default="edited_text")
    p.add_argument("--judge2", help="silent_removal_controls_judge2.csv (Mistral re-run tiers)")
    p.add_argument("--judge2-id", default="base_note_id"); p.add_argument("--judge2-tier", default="validity_tier")
    p.add_argument("--where", action="append", default=[], metavar="COL=VALUE",
                   help="keep only clean-note rows where COL equals VALUE (repeatable, case-insensitive)")
    p.add_argument("--out", default="notes.jsonl")
    a = p.parse_args()

    if a.show_headers:
        for path in a.show_headers:
            rows = read_csv(path)
            print(f"{path}: {len(rows)} rows\n  columns: {list(rows[0].keys()) if rows else '[]'}")
        return

    def need(rows, col, path):
        if rows and col not in rows[0]:
            sys.exit(f"column '{col}' not in {path}; columns are {list(rows[0].keys())}")

    notes = read_csv(a.notes)
    for cond in a.where:
        col, val = cond.split("=", 1)
        need(notes, col, a.notes)
        notes = [r for r in notes if r[col].strip().lower() == val.strip().lower()]
    notes = [r for r in notes if r.get(a.notes_gold, "").strip()]  # must have a gold relation
    for c in (a.notes_id, a.notes_text, a.notes_gold):
        need(notes, c, a.notes)
    rem = {r[a.removal_id]: r for r in read_csv(a.removal)} if a.removal else {}
    keep = {r[a.keep_id]: r for r in read_csv(a.keep)} if a.keep else {}
    j2 = {r[a.judge2_id]: r for r in read_csv(a.judge2)} if a.judge2 else {}

    n_rem = n_keep = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for r in notes:
            nid = r[a.notes_id]
            rr, kk = rem.get(nid), keep.get(nid)
            rec = {
                "note_id": nid,
                "gold": r[a.notes_gold].strip().upper().replace("_", "-"),
                "clean": r[a.notes_text],
                "silent_removal": rr[a.removal_text] if rr and rr.get(a.removal_text) else None,
                "silent_keep": kk[a.keep_text] if kk and kk.get(a.keep_text) else None,
                "orig_removal_tier": (rr.get(a.removal_tier) or None) if rr and a.removal_tier else None,
                "mistral_rerun_tier": (j2[nid].get(a.judge2_tier) or None) if nid in j2 else None,
            }
            n_rem += rec["silent_removal"] is not None
            n_keep += rec["silent_keep"] is not None
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{len(notes)} notes, {n_rem} silent_removal edits (paper: 507), {n_keep} silent_keep edits (paper: 452) -> {a.out}")


if __name__ == "__main__":
    main()
