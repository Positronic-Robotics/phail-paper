"""
PhAIL benchmark: camera/tote position analysis.

Analyzes completion rates for same-side vs opposite-side camera/tote configurations
across all 4 models (OpenPI, GR00T rot6d_rel, ACT, SmolVLA).
"""

import json
import os
from collections import defaultdict

import numpy as np


BASE = os.path.expanduser("~/.cache/positronic/s3/_/inference/phail_final")

# Model name mapping from server.type + modality_config
MODEL_NAMES = {
    "openpi": "OpenPI (pi0.5)",
    "groot:ee_rot6d_rel": "GR00T rot6d_rel",
    "act": "ACT",
    "smolvla": "SmolVLA",
}

# Only include these model keys (skip groot:ee_rot6d)
INCLUDE_MODELS = set(MODEL_NAMES.keys())


def load_episodes():
    """Load all episodes from all batch directories."""
    episodes = []

    # Collect all batch directories
    batch_paths = []

    for item in sorted(os.listdir(BASE)):
        item_path = os.path.join(BASE, item)
        if not os.path.isdir(item_path) or item.startswith("run_"):
            continue

        # Main batch: BASE/000000000000/EPISODE_ID/static.json
        # Date-coded: BASE/DDMMYY/000000000000/EPISODE_ID/static.json
        if item == "000000000000":
            batch_paths.append(item_path)
        else:
            nested = os.path.join(item_path, "000000000000")
            if os.path.isdir(nested):
                batch_paths.append(nested)

    for batch_path in batch_paths:
        for ep_name in sorted(os.listdir(batch_path)):
            static_path = os.path.join(batch_path, ep_name, "static.json")
            if not os.path.isfile(static_path):
                continue

            with open(static_path) as f:
                d = json.load(f)

            server_type = d.get("inference.policy.server.type", "UNKNOWN")
            modality = d.get("inference.policy.server.modality_config", "")
            model_key = f"{server_type}:{modality}" if modality else server_type

            if model_key not in INCLUDE_MODELS:
                continue

            tote = d.get("eval.tote_placement")
            camera = d.get("eval.external_camera")
            successful = d.get("eval.successful_items", 0)
            total = d.get("eval.total_items", 0)

            if tote is None or camera is None or total is None or total == 0:
                continue

            completion = successful / total

            same_side = (tote == camera)

            episodes.append({
                "model_key": model_key,
                "model_name": MODEL_NAMES[model_key],
                "tote": tote,
                "camera": camera,
                "same_side": same_side,
                "successful": successful,
                "total": total,
                "completion": completion,
                "batch": os.path.basename(os.path.dirname(batch_path))
                    if "000000000000" in batch_path else "main",
                "episode": ep_name,
            })

    return episodes


def analyze(episodes):
    # Group by model
    by_model = defaultdict(list)
    for ep in episodes:
        by_model[ep["model_key"]].append(ep)

    # Print counts
    print("=" * 70)
    print("EPISODE COUNTS PER MODEL")
    print("=" * 70)
    for key in ["openpi", "groot:ee_rot6d_rel", "act", "smolvla"]:
        eps = by_model[key]
        print(f"  {MODEL_NAMES[key]:20s}: {len(eps)} episodes")
    print()

    # Per-model analysis
    print("=" * 70)
    print("CAMERA/TOTE POSITION ANALYSIS: SAME-SIDE vs OPPOSITE-SIDE")
    print("=" * 70)
    print()
    print(f"{'Model':20s} | {'Same-side':>10s} | {'Opp-side':>10s} | {'Delta':>8s} | {'N(same)':>7s} | {'N(opp)':>7s}")
    print("-" * 70)

    for key in ["openpi", "groot:ee_rot6d_rel", "act", "smolvla"]:
        eps = by_model[key]

        same = [e for e in eps if e["same_side"]]
        opp = [e for e in eps if not e["same_side"]]

        same_completions = np.array([e["completion"] for e in same])
        opp_completions = np.array([e["completion"] for e in opp])

        same_pct = np.mean(same_completions) * 100 if len(same) > 0 else 0
        opp_pct = np.mean(opp_completions) * 100 if len(opp) > 0 else 0
        delta = same_pct - opp_pct

        print(f"  {MODEL_NAMES[key]:18s} | {same_pct:9.1f}% | {opp_pct:9.1f}% | {delta:+7.1f}pp | {len(same):7d} | {len(opp):7d}")

    print()

    # Detailed GR00T analysis
    print("=" * 70)
    print("GR00T rot6d_rel: DETAILED BREAKDOWN")
    print("=" * 70)
    groot_eps = by_model["groot:ee_rot6d_rel"]

    configs = defaultdict(list)
    for e in groot_eps:
        config = f"tote={e['tote']}, cam={e['camera']}"
        configs[config].append(e)

    for config in sorted(configs.keys()):
        eps = configs[config]
        completions = np.array([e["completion"] for e in eps])
        pct = np.mean(completions) * 100
        print(f"  {config:30s}: {pct:5.1f}% completion ({len(eps)} episodes)")

    same = [e for e in groot_eps if e["same_side"]]
    opp = [e for e in groot_eps if not e["same_side"]]
    same_pct = np.mean([e["completion"] for e in same]) * 100
    opp_pct = np.mean([e["completion"] for e in opp]) * 100
    print()
    print(f"  Same-side:     {same_pct:.1f}% ({len(same)} episodes)")
    print(f"  Opposite-side: {opp_pct:.1f}% ({len(opp)} episodes)")
    print(f"  Delta:         {same_pct - opp_pct:+.1f} percentage points")
    print()

    # Per-model detailed config breakdown
    print("=" * 70)
    print("ALL MODELS: PER-CONFIGURATION BREAKDOWN")
    print("=" * 70)
    for key in ["openpi", "groot:ee_rot6d_rel", "act", "smolvla"]:
        eps = by_model[key]
        print(f"\n  {MODEL_NAMES[key]}:")

        configs = defaultdict(list)
        for e in eps:
            config = f"tote={e['tote']}, cam={e['camera']}"
            configs[config].append(e)

        for config in sorted(configs.keys()):
            config_eps = configs[config]
            completions = np.array([e["completion"] for e in config_eps])
            pct = np.mean(completions) * 100
            side = "SAME" if config_eps[0]["same_side"] else "OPP "
            print(f"    [{side}] {config:30s}: {pct:5.1f}% ({len(config_eps)} eps)")


if __name__ == "__main__":
    episodes = load_episodes()
    print(f"Loaded {len(episodes)} episodes total\n")
    analyze(episodes)
