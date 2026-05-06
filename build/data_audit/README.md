# Data audit — public PhAIL v1.0 dataset

Pulls every published PhAIL episode and produces a single per-episode index
that the rest of the pipeline (drop-event classifier, etc.) can rely on.

## What's published

The benchmark is hosted at <URL withheld for double-blind review>. The underlying data lives in
public Nebius S3:

```
s3://positronic-public/datasets/phail/v1.0/
├── training/   449 episodes — DROID teleoperation demos used to fine-tune the models on the leaderboard
├── inference/  524 episodes — model evaluation runs (production-variant filtered)
└── human/       40 episodes — human baseline (10 per object, all 8/8)
```

It's also reachable over HTTPS without S3 credentials at
`https://storage.eu-north1.nebius.cloud/positronic-public/datasets/phail/v1.0/`
— that's the URL used by Toloka and the Run explorer videos.

The hosted leaderboard at <<URL withheld>/episodes> is the same dataset:

* Concatenation order is `inference + human + training`, so episode index 0 is
  the first inference episode, 524 the first human episode, 564 the first
  training episode.
* JSON feed: <<URL withheld>/api/episodes> (~170 KB, 1013 rows). Bare
  `urllib.request` gets a 403; pass any non-empty User-Agent.
* Per-episode viewer URL: `<URL withheld>/episode/{idx}` where `{idx}` is the
  zero-based index in the API response.

Episode layout on disk (one directory per episode under `<source>/<shard>/<idx>/`):

| File | Purpose | Sizes |
|---|---|---|
| `static.json` | Per-episode static fields baked at release: `model`, `eval.object`, `eval.successful_items`, `eval.total_items`, `eval.outcome`, `eval.duration`, `eval.tote_placement`, `eval.external_camera`, `inference.policy.server.{type,experiment_name,checkpoint_id,checkpoint_path}`, `task` (full sentence), `equipment`, ... | 50–200 KB |
| `meta.json` | System metadata: `created_ts_ns`, `schema_version`, `writer.{name,version,python,platform,git}` | <1 KB |
| `target_grip.parquet` | Commanded gripper signal (binary close/open) | 200–400 KB |
| `grip.parquet` | Measured gripper width (float) | 30–100 KB |
| `robot_state.q.parquet` | Joint positions, 7-vector | 1–4 MB |
| `robot_state.dq.parquet` | Joint velocities, 7-vector | 1–4 MB |
| `robot_state.ee_pose.parquet` | End-effector pose, 7-vector `[tx,ty,tz,qx,qy,qz,qw]` | 1–4 MB |
| `robot_state.error.parquet` | Robot error signal | small |
| `robot_commands.pose.parquet` | Commanded ee pose | ~1 MB |
| `image.exterior.frames.parquet` | Frame timestamps for the third-person camera | ~90 KB |
| `image.exterior.mp4` | Video of the third-person camera | ~5 MB / 30 s |
| `image.wrist.frames.parquet` | Frame timestamps for the wrist camera (training/inference only) | ~90 KB |
| `image.wrist.mp4` | Video of the wrist camera (training/inference only) | ~5 MB / 30 s |

`human/` episodes carry only the exterior camera (no robot, no wrist mount).

## Reproducing the audit

```bash
# One-time install — pos3 is the only thing we need from the positronic stack.
# (The full positronic package pulls mujoco, which fails without MUJOCO_PATH;
#  we don't need it for the audit.)
pip install 'pos3>=0.2.0' pyarrow

# Download metadata + telemetry (~8 GB, no MP4s) and rebuild the index.
python -m build.data_audit.build_index --out-dir out/data_audit

# Cross-check our index against the live leaderboard.
python -m build.data_audit.verify
```

`build_index.py` is idempotent: `pos3.download(..., delete=False)` is a no-op
when the local cache (`~/.cache/positronic/s3/positronic-public/...`) is in
sync. To force a re-download, delete that cache directory.

## Outputs

Files end up under `out/data_audit/` and are *not* committed:

* `episodes.json` — list of 1013 records, one per episode. Schema:
  ```jsonc
  {
    "source":          "training" | "inference" | "human",
    "rel_path":        "000000000000/000000000017",
    "s3_dir":          "s3://positronic-public/datasets/phail/v1.0/...",
    "https_dir":       "https://storage.eu-north1.nebius.cloud/...",
    "public_url":      "<URL withheld>/episode/541",
    "model":           "Physical Intelligence Open π₀.₅",
    "raw_model_type":  "openpi",          // empty for baselines
    "variant":         "270226-ee_rot6d_rel:150000",  // empty for baselines
    "eval.object":     "Wooden spoons",
    "eval.tote_placement":  "left" | "right" | "",
    "eval.external_camera": "left" | "right" | "",
    "eval.outcome":    "Success" | "Fail" | "Safety" | "",
    "status":          "Pass" | "Fail" | "Safety",
    "eval.successful_items": 8,
    "eval.total_items":      8,
    "eval.duration_s":       193.25,      // null for baselines
    "task":            "Pick all the items one by one ...",
    "created_ts_ns":   1774342697081461155,
    "created_iso":     "2026-03-24T08:58:17.081461+00:00",
    "video_streams":   ["exterior", "wrist"]   // ["exterior"] for human/
  }
  ```
* `summary.json` — counts, success rates, tasks, by model and by source.
* `episodes.csv` — same data as `episodes.json` flattened (lists become
  pipe-separated strings) for spreadsheet inspection.
* `build_index.log` — `tee`'d output of the last build run (large; gitignored).

## Latest snapshot (2026-04-29)

| Source | Count |
|---|---:|
| training (DROID teleop) | 449 |
| inference (model eval) | 524 |
| human baseline | 40 |
| **Total** | **1013** |

| Task | Count |
|---|---:|
| Pick-and-place: Wooden spoons | 300 |
| Pick-and-place: Towels | 261 |
| Pick-and-place: Batteries | 237 |
| Pick-and-place: Scissors | 215 |

| Status | Count |
|---|---:|
| Pass | 520 |
| Fail | 471 |
| Safety | 22 |

### Per-model leaderboard

| Model | Episodes | Pass | Fail | Safety | Episode pass | Items succ./total | Item-level success |
|---|---:|---:|---:|---:|---:|---:|---:|
| Physical Intelligence Open π₀.₅ (OpenPI) | 129 | 13 | 112 | 4 | 10.1% | 493/996 | 49.5% |
| NVIDIA GR00T N1.6 | 128 | 10 | 115 | 3 | 7.8% | 458/952 | 48.1% |
| Action Chunking Transformer (ACT) | 151 | 8 | 143 | 0 | 5.3% | 369/1072 | 34.4% |
| Hugging Face SmolVLA | 116 | 0 | 101 | 15 | 0.0% | 92/852 | 10.8% |
| Robot teleoperated by Human | 449 | 449 | 0 | 0 | 100% | 4496/4496 | 100% |
| Human | 40 | 40 | 0 | 0 | 100% | 320/320 | 100% |

Per-task counts per model are in `summary.json`. The four-task split is
near-uniform inside every model except teleop, where Wooden spoons is
over-represented (167/449) — historical artefact of the DROID pool the release
was sourced from.

### Variant filter (already applied)

The `inference/` tree has been **prod-filtered** by the release pipeline:

| `raw_model_type` | Variant kept | Variants dropped |
|---|---|---|
| `groot` | `270226-ee_rot6d_rel:150000` | all others |
| `openpi`, `act`, `smolvla` | (no filter) | — |

So `inference/` size = 524 = 151 ACT + 116 SmolVLA + 128 GR00T + 129 OpenPI;
the leaderboard you see at <URL withheld for double-blind review> is exactly this slice.

### Caveats / known imperfections

* `eval.duration_s` is **null** for `training/` and `human/` baselines. Their
  release transforms set `eval.outcome=Success` but don't provide a duration —
  the actual episode length is implicit in the parquet timestamps
  (`meta.last_ts - meta.start_ts`). Use it from the signal data when needed.
* `eval.tote_placement` and `eval.external_camera` are populated only for
  `inference/` — baselines weren't recorded with the workspace tagging that the
  evaluation runs use.
* The leaderboard JSON is a moving target (re-orders, new submissions). Re-run
  `build_index.py` whenever you need fresh URLs. `verify.py` will exit non-zero
  if our local data drifts from what's published.

## Working with the data downstream

```python
import json
from pathlib import Path

# Records (1013) are sorted by (source, rel_path) — the order on disk.
records = json.loads(Path('out/data_audit/episodes.json').read_text())

# Open an episode using the positronic library:
from positronic.dataset.local_dataset import LocalDataset
ds = LocalDataset(Path.home() / '.cache/positronic/s3/positronic-public/positronic-public/datasets/phail/v1.0/inference')
ep = ds[0]                       # an Episode wired to disk
print(ep['target_grip'])         # commanded gripper signal
print(ep['robot_state.ee_pose']) # 7-vector pose signal
print(ep['eval.object'], ep['eval.outcome'])

# Or address a specific public episode by its public leaderboard index:
url = '<URL withheld>/episode/541'
rec = next(r for r in records if r['public_url'] == url)
print(rec['source'], rec['rel_path'], rec['model'])
```

The existing `build/autopilot/` pipeline already reads `target_grip`, `grip`,
and `robot_state.ee_pose` for drop-event detection — this index gives it a
reliable iteration order plus URLs to surface in any QA UI.
