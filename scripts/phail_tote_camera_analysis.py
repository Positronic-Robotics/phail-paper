"""
PhAIL evaluation: 4-way breakdown by tote position x camera position.
Disentangles tote effect from camera effect from same-side effect.
"""

import json
import glob
import os
from collections import defaultdict

import pos3


BASE = os.path.expanduser("~/.cache/positronic/s3/_/inference/phail_final")

# -- Load all episodes -------------------------------------------------------
episodes = []
with pos3.mirror():
    for path in glob.glob(f"{BASE}/**/static.json", recursive=True):
        d = json.load(open(path))
        tote = d.get("eval.tote_placement")
        cam = d.get("eval.external_camera")
        if not tote or not cam:
            continue

        model_type = d.get("inference.policy.server.type", "unknown")
        exp_name = d.get("inference.policy.server.experiment_name", "unknown")

        # Normalise model names
        if model_type == "openpi":
            model = "pi0.5"
        elif model_type == "groot":
            if "rel" in exp_name:
                model = "GR00T-rel"
            else:
                model = "GR00T"
        elif model_type == "act":
            model = "ACT"
        elif model_type == "smolvla":
            model = "SmolVLA"
        else:
            model = model_type

        episodes.append({
            "model": model,
            "tote": tote,       # left / right
            "camera": cam,      # left / right
            "outcome": d.get("eval.outcome", ""),
            "successful": d.get("eval.successful_items", 0),
            "total": d.get("eval.total_items", 0),
            "duration": d.get("eval.duration", 0),
            "task": d.get("eval.object", ""),
            "path": path,
        })

print(f"Loaded {len(episodes)} episodes\n")

# -- Per-model 4-way table ---------------------------------------------------
CONFIGS = [
    ("TL/CL", "left",  "left"),
    ("TL/CR", "left",  "right"),
    ("TR/CL", "right", "left"),
    ("TR/CR", "right", "right"),
]

models = sorted(set(e["model"] for e in episodes))

# Store aggregate metrics for the summary analysis
agg = {}  # (model, config_label) -> {n, compl, uph, fails}


def compute_metrics(eps_list):
    n = len(eps_list)
    if n == 0:
        return {"n": 0, "compl": 0, "uph": 0, "fails": 0,
                "sum_succ": 0, "sum_total": 0, "sum_dur": 0}
    sum_succ = sum(e["successful"] for e in eps_list)
    sum_total = sum(e["total"] for e in eps_list)
    sum_dur = sum(e["duration"] for e in eps_list)
    compl = (sum_succ / sum_total * 100) if sum_total > 0 else 0
    uph = (sum_succ / (sum_dur / 3600)) if sum_dur > 0 else 0
    fails = sum(1 for e in eps_list if e["outcome"] != "Success")
    return {"n": n, "compl": compl, "uph": uph, "fails": fails,
            "sum_succ": sum_succ, "sum_total": sum_total, "sum_dur": sum_dur}


for model in models:
    m_eps = [e for e in episodes if e["model"] == model]
    print(f"{'=' * 72}")
    print(f"  {model}  ({len(m_eps)} episodes)")
    print(f"{'=' * 72}")
    print(f"  {'Config':<10s}  {'N':>4s}  {'Compl%':>7s}  {'UPH':>6s}  {'Fails':>5s}  {'Succ/Total':>12s}")
    print(f"  {'-'*10}  {'-'*4}  {'-'*7}  {'-'*6}  {'-'*5}  {'-'*12}")

    for label, tote, cam in CONFIGS:
        subset = [e for e in m_eps if e["tote"] == tote and e["camera"] == cam]
        m = compute_metrics(subset)
        agg[(model, label)] = m
        st = f"{m['sum_succ']}/{m['sum_total']}"
        print(f"  {label:<10s}  {m['n']:4d}  {m['compl']:6.1f}%  {m['uph']:6.1f}  {m['fails']:5d}  {st:>12s}")

    # Totals
    m_all = compute_metrics(m_eps)
    st = f"{m_all['sum_succ']}/{m_all['sum_total']}"
    print(f"  {'TOTAL':<10s}  {m_all['n']:4d}  {m_all['compl']:6.1f}%  {m_all['uph']:6.1f}  {m_all['fails']:5d}  {st:>12s}")
    print()


# -- Summary: disentangling effects ------------------------------------------
print(f"\n{'=' * 72}")
print("  DISENTANGLING TOTE vs CAMERA vs SAME-SIDE EFFECTS")
print(f"{'=' * 72}\n")


def merge_metrics(*config_labels, model=None):
    """Combine episodes from multiple configs for a single model."""
    m_eps = [e for e in episodes if e["model"] == model]
    subset = []
    for label, tote, cam in CONFIGS:
        if label in config_labels:
            subset.extend(e for e in m_eps if e["tote"] == tote and e["camera"] == cam)
    return compute_metrics(subset)


def fmt_m(m):
    return f"n={m['n']:3d}  compl={m['compl']:5.1f}%  UPH={m['uph']:5.1f}"


for model in models:
    print(f"  --- {model} ---")

    # 1. Isolate camera effect (hold tote constant)
    tl_cl = agg[(model, "TL/CL")]
    tl_cr = agg[(model, "TL/CR")]
    tr_cl = agg[(model, "TR/CL")]
    tr_cr = agg[(model, "TR/CR")]

    print(f"  Camera effect (tote held constant):")
    print(f"    Tote-Left:   CL={tl_cl['compl']:5.1f}%  vs  CR={tl_cr['compl']:5.1f}%   delta={tl_cl['compl']-tl_cr['compl']:+.1f}pp")
    print(f"    Tote-Right:  CL={tr_cl['compl']:5.1f}%  vs  CR={tr_cr['compl']:5.1f}%   delta={tr_cl['compl']-tr_cr['compl']:+.1f}pp")

    # Average camera effect
    cam_effect = ((tl_cl["compl"] - tl_cr["compl"]) + (tr_cl["compl"] - tr_cr["compl"])) / 2
    print(f"    Avg camera effect (CL minus CR): {cam_effect:+.1f}pp")

    # 2. Isolate tote effect (hold camera constant)
    print(f"  Tote effect (camera held constant):")
    print(f"    Camera-Left:  TL={tl_cl['compl']:5.1f}%  vs  TR={tr_cl['compl']:5.1f}%   delta={tl_cl['compl']-tr_cl['compl']:+.1f}pp")
    print(f"    Camera-Right: TL={tl_cr['compl']:5.1f}%  vs  TR={tr_cr['compl']:5.1f}%   delta={tl_cr['compl']-tr_cr['compl']:+.1f}pp")

    tote_effect = ((tl_cl["compl"] - tr_cl["compl"]) + (tl_cr["compl"] - tr_cr["compl"])) / 2
    print(f"    Avg tote effect (TL minus TR): {tote_effect:+.1f}pp")

    # 3. Same-side vs opposite-side
    same = merge_metrics("TL/CL", "TR/CR", model=model)
    oppo = merge_metrics("TL/CR", "TR/CL", model=model)
    print(f"  Same-side (TL/CL+TR/CR): {fmt_m(same)}")
    print(f"  Opposite  (TL/CR+TR/CL): {fmt_m(oppo)}")
    print(f"    Same minus Opposite:  {same['compl']-oppo['compl']:+.1f}pp completion,  {same['uph']-oppo['uph']:+.1f} UPH")

    # 4. Same-side after controlling for tote
    # If same-side is real, it should appear in BOTH tote positions:
    #   Tote-Left:  CL (same) should beat CR (opposite)
    #   Tote-Right: CR (same) should beat CL (opposite)
    ss_tl = tl_cl["compl"] - tl_cr["compl"]  # same-side advantage when tote=left
    ss_tr = tr_cr["compl"] - tr_cl["compl"]  # same-side advantage when tote=right
    print(f"  Same-side advantage controlled for tote:")
    print(f"    Tote-Left:  same(CL) - opp(CR) = {ss_tl:+.1f}pp")
    print(f"    Tote-Right: same(CR) - opp(CL) = {ss_tr:+.1f}pp")
    if ss_tl > 2 and ss_tr > 2:
        print(f"    --> Consistent: same-side effect appears real in both tote positions")
    elif ss_tl < -2 and ss_tr < -2:
        print(f"    --> Consistent: opposite-side is actually better in both tote positions")
    elif (ss_tl > 2) != (ss_tr > 2):
        direction = "tote-left" if ss_tl > ss_tr else "tote-right"
        print(f"    --> Inconsistent: 'same-side' effect only appears for one tote position")
        print(f"       Likely confounded with tote or camera main effect")
    else:
        print(f"    --> Weak: neither condition shows > 2pp same-side advantage")

    # UPH version
    ss_tl_uph = tl_cl["uph"] - tl_cr["uph"]
    ss_tr_uph = tr_cr["uph"] - tr_cl["uph"]
    print(f"  Same-side UPH advantage controlled for tote:")
    print(f"    Tote-Left:  same(CL) - opp(CR) = {ss_tl_uph:+.1f} UPH")
    print(f"    Tote-Right: same(CR) - opp(CL) = {ss_tr_uph:+.1f} UPH")

    print()


# -- Cross-model summary table ------------------------------------------------
print(f"\n{'=' * 72}")
print("  CROSS-MODEL SUMMARY: EFFECT SIZES (completion pp)")
print(f"{'=' * 72}")
print(f"  {'Model':<12s}  {'Tote(TL-TR)':>12s}  {'Cam(CL-CR)':>12s}  {'SameSide':>10s}  {'SS@TL':>8s}  {'SS@TR':>8s}")
print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*8}")

for model in models:
    tl_cl = agg[(model, "TL/CL")]
    tl_cr = agg[(model, "TL/CR")]
    tr_cl = agg[(model, "TR/CL")]
    tr_cr = agg[(model, "TR/CR")]

    tote_eff = ((tl_cl["compl"] - tr_cl["compl"]) + (tl_cr["compl"] - tr_cr["compl"])) / 2
    cam_eff = ((tl_cl["compl"] - tl_cr["compl"]) + (tr_cl["compl"] - tr_cr["compl"])) / 2

    same = merge_metrics("TL/CL", "TR/CR", model=model)
    oppo = merge_metrics("TL/CR", "TR/CL", model=model)
    ss_raw = same["compl"] - oppo["compl"]

    ss_tl = tl_cl["compl"] - tl_cr["compl"]
    ss_tr = tr_cr["compl"] - tr_cl["compl"]

    print(f"  {model:<12s}  {tote_eff:+11.1f}  {cam_eff:+11.1f}  {ss_raw:+9.1f}  {ss_tl:+7.1f}  {ss_tr:+7.1f}")


# -- GR00T deep dive ----------------------------------------------------------
print(f"\n{'=' * 72}")
print("  GR00T DEEP DIVE: Is the 'same side' effect real?")
print(f"{'=' * 72}\n")

for model in [m for m in models if "GR00T" in m]:
    print(f"  --- {model} ---")
    tl_cl = agg[(model, "TL/CL")]
    tl_cr = agg[(model, "TL/CR")]
    tr_cl = agg[(model, "TR/CL")]
    tr_cr = agg[(model, "TR/CR")]

    print(f"  Raw cell completion rates:")
    print(f"    TL/CL = {tl_cl['compl']:5.1f}%  (n={tl_cl['n']})")
    print(f"    TL/CR = {tl_cr['compl']:5.1f}%  (n={tl_cr['n']})")
    print(f"    TR/CL = {tr_cl['compl']:5.1f}%  (n={tr_cl['n']})")
    print(f"    TR/CR = {tr_cr['compl']:5.1f}%  (n={tr_cr['n']})")

    print(f"\n  Test: Is it just a tote-right effect?")
    tl_avg = (tl_cl["compl"] + tl_cr["compl"]) / 2
    tr_avg = (tr_cl["compl"] + tr_cr["compl"]) / 2
    print(f"    Tote-Left avg:  {tl_avg:.1f}%")
    print(f"    Tote-Right avg: {tr_avg:.1f}%")
    print(f"    Tote effect:    {tl_avg - tr_avg:+.1f}pp")

    cl_avg = (tl_cl["compl"] + tr_cl["compl"]) / 2
    cr_avg = (tl_cr["compl"] + tr_cr["compl"]) / 2
    print(f"    Camera-Left avg:  {cl_avg:.1f}%")
    print(f"    Camera-Right avg: {cr_avg:.1f}%")
    print(f"    Camera effect:    {cl_avg - cr_avg:+.1f}pp")

    ss_tl = tl_cl["compl"] - tl_cr["compl"]
    ss_tr = tr_cr["compl"] - tr_cl["compl"]
    print(f"\n  Same-side advantage (controlled):")
    print(f"    When tote=left:  same(CL) - opp(CR) = {ss_tl:+.1f}pp")
    print(f"    When tote=right: same(CR) - opp(CL) = {ss_tr:+.1f}pp")

    if ss_tl > 2 and ss_tr > 2:
        print(f"    VERDICT: Same-side effect is real in both conditions")
    elif (ss_tl > 0) != (ss_tr > 0):
        print(f"    VERDICT: Effect reverses direction " +
              f"{'-- tote-right drives it' if ss_tr < 0 else '-- tote-left drives it'}")
    elif abs(ss_tl) < 3 and abs(ss_tr) < 3:
        print(f"    VERDICT: No meaningful same-side effect (both < 3pp)")
    else:
        print(f"    VERDICT: Weak/mixed evidence")

    # UPH analysis
    print(f"\n  UPH by cell:")
    print(f"    TL/CL = {tl_cl['uph']:5.1f}  (n={tl_cl['n']})")
    print(f"    TL/CR = {tl_cr['uph']:5.1f}  (n={tl_cr['n']})")
    print(f"    TR/CL = {tr_cl['uph']:5.1f}  (n={tr_cl['n']})")
    print(f"    TR/CR = {tr_cr['uph']:5.1f}  (n={tr_cr['n']})")
    print()


# -- Episode count balance check -----------------------------------------------
print(f"\n{'=' * 72}")
print("  BALANCE CHECK: episode counts per cell")
print(f"{'=' * 72}")
print(f"  {'Model':<12s}  {'TL/CL':>6s}  {'TL/CR':>6s}  {'TR/CL':>6s}  {'TR/CR':>6s}  {'Total':>6s}")
print(f"  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
for model in models:
    cells = [agg[(model, label)]["n"] for label, _, _ in CONFIGS]
    print(f"  {model:<12s}  {cells[0]:6d}  {cells[1]:6d}  {cells[2]:6d}  {cells[3]:6d}  {sum(cells):6d}")
