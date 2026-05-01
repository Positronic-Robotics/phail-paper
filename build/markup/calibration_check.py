"""Compare manual ground-truth markup against release_classifier auto candidates.

Run:
    uv run python -m build.markup.calibration_check

Outputs a summary table to stdout and writes a markdown table to
``build/markup/out/calibration_check.md``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOT_ROOT = REPO_ROOT / "build" / "markup" / "annotations" / "inference"
CANDIDATES_PATH = REPO_ROOT / "out" / "release_classifier" / "candidates.jsonl"
STATIC_CACHE_ROOT = Path(
    "/Users/vertix/.cache/positronic/s3/positronic-public/positronic-public"
    "/datasets/phail/v1.0/inference"
)
OUT_MD = REPO_ROOT / "build" / "markup" / "out" / "calibration_check.md"

TAU_MATCH_S = 3.0
BIG = 1e6  # finite stand-in for infinity in the cost matrix


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_reviewed_rel_ids() -> dict[str, dict[str, str]]:
    """Return mapping rel_id -> {'object': ..., 'model': ...} for reviewed
    inference episodes (any object). The Batteries-only filter is dropped
    so the calibration check covers all four objects once they're annotated."""
    out: dict[str, dict[str, str]] = {}
    for fp in sorted(ANNOT_ROOT.rglob("*.json")):
        with fp.open() as f:
            ann = json.load(f)
        if not ann.get("reviewed"):
            continue
        rel_id = "/".join(fp.relative_to(ANNOT_ROOT).with_suffix("").parts)
        static_p = STATIC_CACHE_ROOT / rel_id / "static.json"
        if not static_p.exists():
            continue
        with static_p.open() as f:
            static = json.load(f)
        out[rel_id] = {
            "object": static.get("eval.object") or "?",
            "model": static.get("model") or "unknown",
        }
    return out


def load_annotation(rel_id: str) -> dict:
    fp = ANNOT_ROOT / f"{rel_id}.json"
    with fp.open() as f:
        return json.load(f)


def load_candidates(rel_ids: Iterable[str]) -> dict[str, list[dict]]:
    """Load candidates.jsonl filtered to inference rel_paths in `rel_ids`.
    No object filter — rel_path alone uniquely identifies the episode within
    the inference source, and we want all four objects covered."""
    keep = set(rel_ids)
    by_rel: dict[str, list[dict]] = defaultdict(list)
    with CANDIDATES_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("source") != "inference":
                continue
            rel = r.get("rel_path")
            if rel not in keep:
                continue
            by_rel[rel].append(r)
    return by_rel


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def hungarian_match(
    manual_ts: list[float], auto_ts: list[float], tau: float = TAU_MATCH_S
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Hungarian 1-1 matching with τ cutoff.

    Returns
    -------
    matched : list of (i_manual, j_auto, |Δt|)
    manual_unmatched : list of manual indices
    auto_unmatched : list of auto indices
    """
    m, n = len(manual_ts), len(auto_ts)
    if m == 0 and n == 0:
        return [], [], []
    if m == 0:
        return [], [], list(range(n))
    if n == 0:
        return [], list(range(m)), []

    cost = np.full((m, n), BIG, dtype=float)
    for i, mt in enumerate(manual_ts):
        for j, at in enumerate(auto_ts):
            d = abs(mt - at)
            if d <= tau:
                cost[i, j] = d
    row_ind, col_ind = linear_sum_assignment(cost)
    matched: list[tuple[int, int, float]] = []
    used_m: set[int] = set()
    used_a: set[int] = set()
    for i, j in zip(row_ind, col_ind):
        c = cost[i, j]
        if c < BIG:
            matched.append((int(i), int(j), float(c)))
            used_m.add(int(i))
            used_a.add(int(j))
    manual_unmatched = [i for i in range(m) if i not in used_m]
    auto_unmatched = [j for j in range(n) if j not in used_a]
    return matched, manual_unmatched, auto_unmatched


# ---------------------------------------------------------------------------
# Per-episode analysis
# ---------------------------------------------------------------------------

def analyze_episode(ann: dict, cands: list[dict]) -> dict:
    place_ts = list(ann.get("place_timestamps_s") or [])
    drop_ts = list(ann.get("drop_timestamps_s") or [])

    succ_cands = [c for c in cands if c["features"].get("verdict") == "success"]
    fail_cands = [c for c in cands if c["features"].get("verdict") == "failure"]
    neut_cands = [c for c in cands if c["features"].get("verdict") == "neutral"]

    succ_ts = [c["ts_s"] for c in succ_cands]
    fail_ts = [c["ts_s"] for c in fail_cands]
    all_ts = [c["ts_s"] for c in cands]

    # Stage-1 RECALL: for each manual placement, is there ANY candidate (any
    # verdict) within τ? If yes but it's not "success", it's a stage-2 mis-class.
    matched_succ, manual_unm_succ, auto_unm_succ = hungarian_match(place_ts, succ_ts)
    matched_any, manual_unm_any, _ = hungarian_match(place_ts, all_ts)

    # Drops
    matched_drop, manual_unm_drop, auto_unm_drop = hungarian_match(drop_ts, fail_ts)

    # Stage-1 miss: manual placement with NO candidate within τ at all.
    # Stage-2 mis-class: manual placement matched to a candidate, but that
    # candidate's verdict was not "success".
    n_manual = len(place_ts)
    n_matched_succ = len(matched_succ)
    n_matched_any = len(matched_any)
    n_stage1_miss = len(manual_unm_any)
    n_matched_other = n_matched_any - n_matched_succ

    return {
        "n_manual_places": n_manual,
        "n_auto_success": len(succ_cands),
        "n_auto_failure": len(fail_cands),
        "n_auto_neutral": len(neut_cands),
        "n_matched_succ": n_matched_succ,
        "n_matched_other": n_matched_other,
        "n_stage1_miss": n_stage1_miss,
        "n_auto_succ_unmatched": len(auto_unm_succ),
        "matched_succ_dts": [c for _, _, c in matched_succ],
        "n_manual_drops": len(drop_ts),
        "n_auto_failures": len(fail_cands),
        "n_drop_matched": len(matched_drop),
        "n_drop_manual_only": len(manual_unm_drop),
        "n_drop_auto_only": len(auto_unm_drop),
        "n_neutral": len(neut_cands),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def aggregate(per_ep: dict[str, dict], rel_ids: Iterable[str]) -> dict:
    keys_int = [
        "n_manual_places",
        "n_auto_success",
        "n_auto_failure",
        "n_auto_neutral",
        "n_matched_succ",
        "n_matched_other",
        "n_stage1_miss",
        "n_auto_succ_unmatched",
        "n_manual_drops",
        "n_auto_failures",
        "n_drop_matched",
        "n_drop_manual_only",
        "n_drop_auto_only",
        "n_neutral",
    ]
    agg = {k: 0 for k in keys_int}
    dts: list[float] = []
    n_eps = 0
    for rel in rel_ids:
        ep = per_ep[rel]
        n_eps += 1
        for k in keys_int:
            agg[k] += ep[k]
        dts.extend(ep["matched_succ_dts"])
    agg["n_eps"] = n_eps
    agg["matched_succ_dts"] = dts
    return agg


def fmt_pct(num: int, den: int) -> str:
    if den == 0:
        return "n/a"
    return f"{num} ({100.0 * num / den:.1f}%)"


def render_cohort(label: str, agg: dict) -> str:
    n_manual = agg["n_manual_places"]
    n_succ = agg["n_auto_success"]
    dts = sorted(agg["matched_succ_dts"])
    if dts:
        med = float(np.median(dts))
        p90 = float(np.percentile(dts, 90))
        mx = float(max(dts))
        time_block = (
            f"  median |Δt|: {med:.3f} s\n"
            f"  90th pct:    {p90:.3f} s\n"
            f"  max:         {mx:.3f} s\n"
        )
    else:
        time_block = "  (no matched success pairs)\n"

    lines = [
        f"COHORT: {label} (n_eps={agg['n_eps']})",
        "",
        "Stage-1 RECALL (success placements):",
        f"  manual placements:        {n_manual}",
        f"  matched to auto-success:  {fmt_pct(agg['n_matched_succ'], n_manual)}",
        f"  matched to auto-other:    {fmt_pct(agg['n_matched_other'], n_manual)}"
        "   [ts in candidates.jsonl, but verdict was failure/neutral]",
        f"  unmatched (stage-1 miss): {fmt_pct(agg['n_stage1_miss'], n_manual)}",
        "",
        "Stage-2 SUCCESS PRECISION:",
        f"  auto-success calls:       {n_succ}",
        f"  matched to manual:        {fmt_pct(agg['n_matched_succ'], n_succ)}",
        f"  unmatched (false pos):    {fmt_pct(agg['n_auto_succ_unmatched'], n_succ)}",
        "",
        "Drop alignment (manual drop_timestamps_s vs auto verdict==failure):",
        f"  manual drops:             {agg['n_manual_drops']}",
        f"  auto failures:            {agg['n_auto_failures']}",
        f"  matched:                  {agg['n_drop_matched']}",
        f"  manual-only:              {agg['n_drop_manual_only']}",
        f"  auto-only:                {agg['n_drop_auto_only']}",
        "",
        "Time error (matched success pairs):",
        time_block.rstrip(),
        "",
        "NEUTRAL candidates (per the schema, these shouldn't match anything):",
        f"  total:                    {agg['n_neutral']}",
    ]
    return "\n".join(lines)


def render_per_episode(per_ep: dict[str, dict], meta: dict[str, dict[str, str]]) -> str:
    rows = []
    for rel, ep in per_ep.items():
        rows.append({
            "rel_id": rel,
            "object": meta[rel]["object"],
            "model": meta[rel]["model"],
            "n_manual_places": ep["n_manual_places"],
            "n_auto_success": ep["n_auto_success"],
            "n_matched": ep["n_matched_succ"],
            "n_manual_unmatched": ep["n_stage1_miss"] + ep["n_matched_other"],
            "n_auto_unmatched": ep["n_auto_succ_unmatched"],
        })
    rows.sort(key=lambda r: (-r["n_manual_unmatched"], -r["n_auto_unmatched"]))
    header = (
        "| rel_id | object | model | n_manual | n_auto_succ | n_matched | "
        "n_manual_unm | n_auto_unm |\n"
        "|---|---|---|---:|---:|---:|---:|---:|"
    )
    body = "\n".join(
        f"| {r['rel_id']} | {r['object']} | {r['model']} | {r['n_manual_places']} | "
        f"{r['n_auto_success']} | {r['n_matched']} | "
        f"{r['n_manual_unmatched']} | {r['n_auto_unmatched']} |"
        for r in rows
    )
    return header + "\n" + body


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    meta = load_reviewed_rel_ids()
    rel_ids = sorted(meta)
    cands_by_rel = load_candidates(rel_ids)

    per_ep: dict[str, dict] = {}
    for rel in rel_ids:
        ann = load_annotation(rel)
        cands = cands_by_rel.get(rel, [])

        # Sanity: if candidate_annotations is present, len(place_ts) must equal
        # sum(n_successful).
        if "candidate_annotations" in ann:
            n_marked = sum(int(c.get("n_successful", 0)) for c in ann["candidate_annotations"])
            if len(ann.get("place_timestamps_s") or []) != n_marked:
                print(
                    f"  WARN {rel}: place_timestamps_s ({len(ann['place_timestamps_s'])}) "
                    f"!= sum n_successful ({n_marked})"
                )

        per_ep[rel] = analyze_episode(ann, cands)

    pooled = aggregate(per_ep, rel_ids)

    # Split by object (the more interesting axis given recall-is-failing-per-
    # object hypothesis), then by model within each object.
    by_object: dict[str, list[str]] = defaultdict(list)
    by_model: dict[str, list[str]] = defaultdict(list)
    by_object_model: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rel, m in meta.items():
        by_object[m["object"]].append(rel)
        by_model[m["model"]].append(rel)
        by_object_model[(m["object"], m["model"])].append(rel)

    sections = []
    label = (
        f"inference (all reviewed objects) — pooled (n_eps={len(rel_ids)}, "
        + ", ".join(f"{o}={len(v)}" for o, v in sorted(by_object.items()))
        + ")"
    )
    sections.append(render_cohort(label, pooled))
    for obj in sorted(by_object):
        rels = sorted(by_object[obj])
        per_model_breakdown = ", ".join(
            f"{m}={sum(1 for r in rels if meta[r]['model'] == m)}"
            for m in sorted({meta[r]["model"] for r in rels})
        )
        sections.append(render_cohort(
            f"inference {obj} (per-model: {per_model_breakdown})",
            aggregate(per_ep, rels),
        ))

    per_ep_md = render_per_episode(per_ep, meta)

    output_md_parts = [
        "# Calibration check: manual markup vs release_classifier auto candidates",
        "",
        f"τ_match = {TAU_MATCH_S}s, Hungarian 1-1 matching.",
        "",
    ]
    for s in sections:
        output_md_parts.append("```")
        output_md_parts.append(s)
        output_md_parts.append("```")
        output_md_parts.append("")
    output_md_parts.append("## Per-episode disagreement (sorted by n_manual_unmatched DESC)")
    output_md_parts.append("")
    output_md_parts.append(per_ep_md)
    output_md_parts.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(output_md_parts))

    # Console output
    for s in sections:
        print(s)
        print()
    print("Per-episode disagreement (top 10 by n_manual_unmatched):")
    print(per_ep_md.split("\n")[0])
    print(per_ep_md.split("\n")[1])
    for line in per_ep_md.split("\n")[2:12]:
        print(line)
    print()
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
