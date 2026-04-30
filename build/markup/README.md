# build/markup — manual markup UI

Side-by-side wrist + exterior camera review of stage-1 release candidates.
Per-candidate keyboard-driven `n_successful` / `n_out_of_bins` / `comment`
inputs; auto-saves to a sidecar JSON outside the source dataset; stage-1
output (`out/release_classifier/candidates.jsonl`) drives the candidate
list when `--candidates` is given.

See `build/WORKFLOW.md` (top-level) for the full end-to-end pipeline
(audit → detect → markup), prerequisites, outcomes, and operator flow.

## Quick start

```bash
uv run python -m build.markup.ui \
  --debug_set build/autopilot/debug_set.json \
  --cache_root ~/.cache/positronic/s3/positronic-public/positronic-public/datasets/phail/v1.0/inference \
  --candidates out/release_classifier/candidates.jsonl
```

## Outputs

`build/markup/annotations/{source}/{rel_id}.json`. The source dataset is
never modified.

## Files

| file | purpose |
|---|---|
| `ui.py` | dearpygui app: cameras, scrubber, candidate review fields, key bindings |
| `io.py` | sidecar read/write, `CandidateAnnotation` dataclass |
| `annotations/` | per-episode JSON sidecars (gitignored) |
