# Build pipeline workflow — release classifier + markup UI

End-to-end recipe for an agent to (re-)run the release-classifier pipeline,
launch the markup UI on the resulting candidates, and iterate. Three stages:
audit → stage 1+2 detector → manual review.

```
build/data_audit/                 build/release_classifier/             build/markup/
  build_index.py        --->        run_all.py             --->           ui.py
  download_videos.py                calibrate_bins.py                     io.py
                                    validate_recall.py
                                    validate_outcome.py
                                    report.py
```

All commands are run from the repo root (`/path/to/phail-paper`). Use the
project's `uv run` (or the matching virtualenv) so `pos3`, `positronic`,
`dearpygui` and `botocore` are available. The detector uses no GPU.

---

## Stage 0 — Build the audit index

**Why first**: every downstream stage joins per-episode metadata (model,
object, S/N, eval.outcome, eval.notes, public_url) from the audit. Without
it `run_all.py` errors out and the markup UI can't show episode headers.

```bash
uv run python -m build.data_audit.build_index
```

### Prerequisites
- Network access to the public S3 bucket
  (`s3://positronic-public/datasets/phail/v1.0/`).
- ~2–3 GB free disk for parquet metadata + static.json under
  `~/.cache/positronic/s3/positronic-public/positronic-public/datasets/phail/v1.0/`.
- No AWS credentials needed: the `PUBLIC_PROFILE` defined in
  `build/data_audit/build_index.py` is anonymous.

### Outcomes
- Local mirror under `~/.cache/positronic/.../v1.0/` with parquets and
  metadata for every episode (no mp4 yet).
- `out/data_audit/episodes.json` — one record per episode, the canonical
  audit index used by stage 1+2 and by the markup UI's per-episode header.
- `out/data_audit/build_index.log` (gitignored).

### Things to look at on completion
- The script prints `Wrote out/data_audit/episodes.json (N episodes)`.
- Per-source counts: should land near 524 inference + 449 training + 40 human.
- If a source is empty, S3 sync was likely interrupted; re-run.

### Optional — pull the videos
The audit/detector don't need mp4s. The markup UI does.

```bash
uv run python -m build.data_audit.download_videos --sources inference training human
```

~26 GB total (`inference` ~16 GB, `training` ~10 GB, `human` smaller). Re-runnable;
already-mirrored files are skipped. Watch the log line — `delete=False` means
it won't remove anything you have locally.

---

## Stage 1+2 — Run the release-classifier pipeline

```bash
# First-time / fresh calibration:
uv run python -m build.release_classifier.run_all                   # writes a stage-1-only candidates.jsonl
uv run python -m build.release_classifier.calibrate_bins            # fits bins.yaml from stage-1 output
uv run python -m build.release_classifier.run_all                   # re-runs with stage-2 verdicts populated

# Subsequent runs (bins.yaml exists):
uv run python -m build.release_classifier.run_all
```

`run_all` is the only end-to-end driver. It walks every episode in the
audit index, runs `detect → featurize → classify` per [pipeline.py],
emits one row per candidate to `candidates.jsonl`, and one row per episode
to `episode_summary.csv`.

### Prerequisites
- `out/data_audit/episodes.json` from stage 0.
- `~/.cache/positronic/.../v1.0/` with parquets present (no mp4s required).
- For stage 2 verdicts to fill in correctly:
  `build/release_classifier/bins.yaml` must exist. If absent, run
  `calibrate_bins` (it reads `out/release_classifier/candidates.jsonl`,
  i.e. needs a prior stage-1 run).

### Outcomes
- `out/release_classifier/candidates.jsonl` — one JSONL line per release
  candidate. Each line carries source, rel_path, audit metadata,
  per-candidate features, and the stage-2 verdict (`success` /
  `failure` / `neutral` / `unknown`). This is the file the markup UI
  consumes.
- `out/release_classifier/episode_summary.csv` — one row per episode
  with S, N, expected_min, n_commanded, n_accidental, status.
- `out/release_classifier/report.html` — generate via
  `python -m build.release_classifier.report`. Static HTML grouping
  episodes by recall status with per-candidate verdict tags + reasons +
  deep-links into the phail.ai viewer.

### Things to look at after run_all
- Final summary line: `Wrote N candidates from M episodes (K had no robot signals)`.
  K should equal the human count (~40). If higher, parquet sync incomplete.
- Run `python -m build.release_classifier.validate_recall` for stage-1
  per-source bound-met %.
- Run `python -m build.release_classifier.validate_outcome` for stage-2
  success/failure recall vs. the audit's eval.outcome.

  Healthy numbers (current detector, v1.0):
  ```
  inference  100% success_recall (audit-Success ≥1 SUCCESS verdict)
  training   100% success_recall
  ```

### Verdict semantics in candidates.jsonl
| verdict | meaning | review action |
|---|---|---|
| `success` | release ee inside target bin | review for `n_successful` count |
| `failure` | release ee outside *any* bin (off-table drop) or robot_error during would-be success | review for `n_out_of_bins` count |
| `neutral` | release ee back in source bin (fumble / re-grip) — no item placed, no item lost | auto-skipped from review queue |
| `unknown` | missing pose / pickup data | review (judge from video) |

### Tuning knobs (if you change them, re-run pipeline + tests)
- `build/release_classifier/detect.py`:
  `HOLD_WINDOW_MIN_MS`, `GRIP_STABILITY_BAND` — stage-1 plateau gate.
- `build/release_classifier/featurize.py`:
  `PICKUP_Z_RISE_M`, `TARGET_RIM_MARGIN_M`, `HOLD_MS_MIN/MAX`,
  `INTER_BIN_DIST_TOLERANCE_*_M` — stage-2 consistency thresholds.
- `build/release_classifier/calibrate_bins.py`: `--margin-m` and the
  percentile band — affects bbox tightness.

### Tests
```bash
uv run python -m pytest build/release_classifier/tests -q
```
should be green (91 tests at last count).

---

## Stage 3 — Manual markup with the UI

```bash
uv run python -m build.markup.ui \
  --debug_set build/autopilot/debug_set.json \
  --cache_root ~/.cache/positronic/s3/positronic-public/positronic-public/datasets/phail/v1.0/inference \
  --candidates out/release_classifier/candidates.jsonl
```

### Prerequisites
- `out/release_classifier/candidates.jsonl` from stage 1+2.
- An episode list. Two ways:
  - `--debug_set <path>` — JSON list of `{ep_dir, model, object, ...}` records.
    The bundled `build/autopilot/debug_set.json` has 50 curated inference
    episodes. **Filter to only-locally-cached episodes** if videos aren't
    fully downloaded:
    ```bash
    uv run python -c '
    import json
    from pathlib import Path
    debug = json.load(open("build/autopilot/debug_set.json"))
    cache = Path.home() / ".cache/positronic/s3/positronic-public/positronic-public/datasets/phail/v1.0/inference"
    ready = [e for e in debug if (cache / e["ep_dir"] / "image.exterior.mp4").exists()
             and (cache / e["ep_dir"] / "image.wrist.mp4").exists()]
    Path("/tmp/markup_ready_set.json").write_text(json.dumps(ready, indent=2))
    print(f"{len(ready)}/{len(debug)} episodes ready")
    '
    # then pass --debug_set /tmp/markup_ready_set.json
    ```
  - `--episode_root <dir>` — walks a directory of `static.json`-bearing
    episode dirs.
- `--cache_root` should point to the on-disk source bucket (e.g.
  `.../v1.0/inference` for inference episodes). The trailing path
  segment is parsed to determine `source` ∈ {`inference`, `training`,
  `human`}; this is critical because rel_ids collide across sources.
- mp4 files for every episode in the set. Trying to load an episode
  without its videos crashes PyAV.

### One-time startup
The UI **preloads every EpisodeView** at launch (opens libav containers
and decodes one frame per stream). Expect ~30–90 s of `preloading X/N`
log lines before the window appears. After that, episode switching is
instantaneous and no further libav containers open during the session.
This avoids a known PyAV crash when opening the 3rd+ container while
prior frames are still decoding.

### Outcomes
- One JSON sidecar per episode under
  `build/markup/annotations/{source}/{rel_id}.json` (e.g.
  `annotations/inference/000000000000/000000000028.json`). Includes:
  ```jsonc
  {
    "place_timestamps_s": [29.987],          // derived: each ts repeated n_successful times
    "items_in_target_at_end": 1,              // = sum(n_successful)
    "items_in_source_at_end": 0,
    "candidate_annotations": [
      {"ts_s": 29.987, "n_successful": 1, "n_out_of_bins": 0, "comment": "..."},
      ...
    ]
  }
  ```
- One annotation per stage-1 candidate that the UI showed. NEUTRAL
  candidates are auto-resolved to `n_successful=0, n_out_of_bins=0`
  without user input but still recorded.
- The source dataset under `~/.cache/.../v1.0/` is **never written to** —
  pos3 is read-only; markup is sidecar-only.

### Things to look at on completion of an annotation pass
- `build/markup/annotations/{source}/000000000000/{NNN}.json` exists
  for every episode you visited.
- `items_in_target_at_end` should match `eval.successful_items` from
  the audit when the policy and audit agree. Mismatches surface for
  per-event diagnosis.

---

## Stage 3.5 — How to use the markup UI (operator flow)

The UI follows a strict **per-candidate review loop**: it seeks 0.5 s
before each non-NEUTRAL candidate, plays through, auto-pauses 0.5 s after
the candidate's timestamp, prompts for `n_successful` / `n_out_of_bins` /
optional `comment`, and on `Enter` advances to the next reviewable
candidate.

### Layout
- **Top**: two cameras side-by-side (exterior left, wrist right).
- **Middle**: scrubber, timeline strip with stage-1 candidates as
  light-blue ticks (current candidate is pink).
- **Status row**: `t=…/duration`, items_in_source / target counters,
  save status (when last persisted).
- **Candidate review block** (only when `--candidates` is given):
  - candidate header: `candidate i/N @ ts.s s  verdict=…  kind=…`
  - **n_successful** input (0–99, default 1 for SUCCESS, 0 otherwise)
  - **n_out_of_bins** input (0–99, default 1 for FAILURE, 0 otherwise)
  - **comment** input (free-form, auto-saves on every keystroke)
  - reasons line: stage-2 verdict reasons + key hint
- **Action buttons**: `Prev cand (Bksp)` · `Save & Next (Enter)` ·
  `Play / Pause (Space)` · `Prev episode (,)` · `Next episode (.)` ·
  `Quit (Q)`. Every button is also a key.
- **Bottom**: KEYBINDINGS section (two columns).

### Per-candidate iteration
1. UI auto-seeks to `ts − 0.5 s`, plays, auto-pauses at `ts + 0.5 s`.
2. The active field (yellow border) is `n_successful` by default.
3. **Type a digit (0–9)** → replaces the active field's value. Skipped
   when the comment field has dearpygui focus (so digits in comments
   work as text).
4. **Tab** → toggles active field between `n_successful` and `n_out_of_bins`.
5. **Click** any field directly to type into it (dearpygui's natural
   input handling — append-style, not replace).
6. **Optional**: type into the `comment` field for free-form notes.
   Comments persist alongside the per-candidate counts and remain
   visible if you revisit the episode later.
7. **Enter** → commit, advance to next *reviewable* candidate
   (NEUTRAL candidates auto-advance through). At the last candidate,
   the prev/next-episode buttons get the active-yellow theme as a hint.
8. **Backspace** → previous reviewable candidate (skips NEUTRAL).
9. **`,` / `.`** → previous / next episode. The cached EpisodeView is
   re-shown if you've visited before; new episodes were preloaded at
   startup so this is decode-free.

### Resumability
- All saves are atomic (`tmp` + `replace`).
- Annotations persist across UI restarts. The UI re-reads the sidecar
  at episode-load and pre-fills any user values you've already entered.
- You can re-run the pipeline with new gate parameters; the UI's
  bootstrap uses verdict to pre-fill new candidate timestamps and
  preserves existing user values for any timestamp that survived.

### Common pitfalls
- **"`.` crashes the UI"** — only happens if you ran the UI before the
  preload-at-startup change (current code preloads all containers up
  front). Symptom: SIGABRT in `av_fast_realloc` / `VideoStream.decode`.
  Fix: `git pull` and run the latest UI.
- **"All my candidates show n_successful=0"** — likely you opened the
  UI without `--candidates`, so the candidate-review block is hidden
  and you're in free-scrub mode. Add `--candidates`.
- **"UI shows candidates that don't match the video"** — you launched
  with a `--cache_root` whose trailing segment doesn't match the
  expected source. The candidates loader keys by `(source, rel_path)`;
  if `--cache_root` ends in `inference`, the UI looks up inference
  candidates. If you mistakenly passed `.../training`, you'll see
  training/ candidates over the inference video.
- **"Missing Dependency: botocore[crt]"** — pos3 default profile tries
  to load real AWS credentials. The UI passes `PUBLIC_PROFILE`
  (anonymous) so this should not happen for the public dataset.

---

## End-to-end one-liner (for an agent)

```bash
# Stage 0
uv run python -m build.data_audit.build_index

# (Optional — only if running the markup UI)
uv run python -m build.data_audit.download_videos

# Stage 1+2 — first time
uv run python -m build.release_classifier.run_all
uv run python -m build.release_classifier.calibrate_bins
uv run python -m build.release_classifier.run_all
uv run python -m build.release_classifier.validate_outcome
uv run python -m build.release_classifier.report

# Stage 3
uv run python -m build.markup.ui \
  --debug_set build/autopilot/debug_set.json \
  --cache_root ~/.cache/positronic/s3/positronic-public/positronic-public/datasets/phail/v1.0/inference \
  --candidates out/release_classifier/candidates.jsonl
```
