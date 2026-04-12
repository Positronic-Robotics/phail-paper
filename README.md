# PhAIL Paper

Draft and analysis scripts for "PhAIL: A Real-Robot Benchmark for Production-Ready Physical AI".

## Data

Scripts read episode data from `~/.cache/positronic/s3/_/inference/phail_final`.
Sync with: `uv run pos3 sync s3://positronic/data/inference/phail_final`

## Running

```bash
uv run scripts/statistical_analysis.py
uv run scripts/uph_analysis.py
uv run scripts/uph_reanalysis.py
uv run scripts/phail_tote_camera_analysis.py
uv run scripts/tote_camera_recheck.py
```
