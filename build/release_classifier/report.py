"""Render out/release_classifier/report.html — a single self-contained static
page grouping episodes by recall status (MISSING first), with each candidate
deep-linking into the public episode viewer at the right timestamp.

The viewer accepts `?t=<seconds>` to open paused at a given offset from the
start of the episode (see positronic_server.py:545). We seek a small
`VIEWER_LEAD_IN_S` *before* the candidate timestamp so the reviewer sees
the run-up to the release rather than the moment after it. Raw MP4 links
are intentionally *not* emitted — downloading multi-megabyte clips for
visual triage is slower and worse than the in-browser viewer.

Inputs:
  - out/release_classifier/episode_summary.csv (from run_all)
  - out/release_classifier/candidates.jsonl    (from run_all)

Output:
  - out/release_classifier/report.html

Usage:
    uv run python -m build.release_classifier.report
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

from .types import FeatureName, RecallStatus, Source, Verdict

CSS = """
body { font: 13px/1.4 -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1280px; margin: 1.5em auto; color: #222; padding: 0 12px; }
h1 { margin: 0 0 .25em 0; }
h2 { margin: 1.5em 0 .25em 0; padding: .25em 0; border-bottom: 2px solid #888; }
h3 { margin: 1em 0 .25em 0; }
section.episode { border: 1px solid #ddd; border-radius: 4px; padding: 8px 10px;
                  margin: 8px 0; background: #fafafa; }
.episode header { display: flex; flex-wrap: wrap; gap: .5em 1.25em; align-items: baseline; }
.episode header .meta { color: #444; font-size: 12px; }
.episode header a { font-size: 12px; }
.tag { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px;
       color: #fff; font-weight: 600; }
.tag.missing { background: #c0392b; }
.tag.match { background: #16a085; }
.tag.extra { background: #d68910; }
.tag.invalid { background: #95a5a6; }
.tag.kind { background: #555; font-weight: 500; }
.tag.kind.commanded_release { background: #2c3e50; }
.tag.kind.accidental_drop { background: #7d3c98; }
.tag.verdict { font-weight: 600; }
.tag.verdict.success { background: #16a085; }
.tag.verdict.failure { background: #c0392b; }
.tag.verdict.unknown { background: #95a5a6; }
.tag.verdict.invalid { background: #95a5a6; }
.reason { display: inline-block; background: #eef; color: #335;
          border-radius: 3px; padding: 0 4px; margin-right: 3px;
          font-size: 11px; }
.reason.fail { background: #fee; color: #933; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th, td { padding: 3px 6px; border-bottom: 1px solid #eee; text-align: right; vertical-align: top; }
th:first-child, td:first-child { text-align: left; }
th { background: #eee; font-weight: 600; }
td.kind, th.kind { text-align: left; }
td.zone, th.zone { text-align: left; }
td.links { text-align: left; white-space: nowrap; }
td.links a { margin-right: .5em; }
.summary-table th { font-weight: 600; }
.summary-table td.num, .summary-table th.num { text-align: right; }
.warn { color: #c0392b; font-weight: 600; }
.muted { color: #888; }
.notes { background: #fffbe6; border-left: 3px solid #f1c40f; padding: 2px 6px;
         margin-top: 4px; font-size: 12px; color: #555; }
.fixed { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
nav { font-size: 12px; margin: .5em 0; }
nav a { margin-right: 1em; }
"""


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------


def _h(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, float) and not math.isfinite(value):
        return '<span class="muted">nan</span>'
    return html.escape(str(value))


def _fmt_float(v: Any, places: int = 3) -> str:
    if v is None:
        return ''
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _h(v)
    if not math.isfinite(f):
        return '<span class="muted">nan</span>'
    return f'{f:.{places}f}'


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _read_candidates(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.setdefault((rec['source'], rec['rel_path']), []).append(rec)
    for cs in out.values():
        cs.sort(key=lambda r: r['ts_ns'])
    return out


# ---------------------------------------------------------------------------
# Per-candidate row building.
# ---------------------------------------------------------------------------

# Open the viewer this many seconds BEFORE the candidate timestamp so the
# user sees the run-up to the release, not the moment after. Tunable; the
# detector emits the falling edge for COMMANDED_RELEASE and the slump
# midpoint for ACCIDENTAL_DROP, both of which benefit from a small lead-in.
VIEWER_LEAD_IN_S: float = 1.5


def _viewer_url(public_url: str, ts_s: float) -> str | None:
    """Build a public viewer URL paused `ts_s` seconds (minus a small
    lead-in) from the start of the episode. The viewer's `?t=` param is
    documented in positronic_server.py.
    """
    if not public_url:
        return None
    seek = max(0.0, ts_s - VIEWER_LEAD_IN_S)
    return f'{public_url}?t={seek:.2f}'


def _candidate_link(public_url: str, ts_s: float) -> str:
    url = _viewer_url(public_url, ts_s)
    if not url:
        return ''
    return f'<a href="{html.escape(url)}" target="_blank">view@{ts_s:.2f}s</a>'


def _candidate_row(c: dict[str, Any], public_url: str) -> str:
    f = c.get('features') or {}
    ts_s = float(c['ts_s'])
    kind = c['kind']
    verdict = f.get(FeatureName.VERDICT.slug) or 'invalid'
    reasons = f.get(FeatureName.VERDICT_REASONS.slug) or []
    is_fail = verdict == Verdict.FAILURE.slug
    reason_chips = ' '.join(
        f'<span class="reason{(" fail" if is_fail else "")}">{html.escape(str(r))}</span>'
        for r in reasons
    )
    err_cell = '<span class="warn">!</span>' if f.get(FeatureName.ROBOT_ERROR_WITHIN_500MS.slug) else ''
    return (
        '<tr>'
        f'<td class="kind"><span class="tag kind {html.escape(kind)}">{html.escape(kind)}</span></td>'
        f'<td><span class="tag verdict {html.escape(verdict)}">{html.escape(verdict).upper()}</span></td>'
        f'<td>{ts_s:.2f}</td>'
        f'<td>{_fmt_float(f.get(FeatureName.PAIRED_CLOSE_TS_S.slug), 2)}</td>'
        f'<td>{_fmt_float(f.get(FeatureName.HOLD_MS.slug), 0)}</td>'
        f'<td>{_fmt_float(f.get(FeatureName.DISPLACEMENT_M.slug), 3)}</td>'
        f'<td class="zone">{_h(f.get(FeatureName.EE_Y_ZONE.slug))}</td>'
        f'<td class="zone">{_h(f.get(FeatureName.PAIRED_CLOSE_Y_ZONE.slug))}</td>'
        f'<td>{_fmt_float(f.get(FeatureName.GRIP_BEFORE_MEAN_150MS.slug), 3)}</td>'
        f'<td>{_fmt_float(f.get(FeatureName.GRIP_AFTER_MEAN_150MS.slug), 3)}</td>'
        f'<td>{_h(f.get(FeatureName.TARGET_ZONE_BALANCE_BEFORE.slug))}</td>'
        f'<td>{err_cell}</td>'
        f'<td class="zone">{reason_chips}</td>'
        f'<td class="links">{_candidate_link(public_url, ts_s)}</td>'
        '</tr>'
    )


# ---------------------------------------------------------------------------
# Per-episode block.
# ---------------------------------------------------------------------------


def _episode_header_meta(row: dict[str, str]) -> str:
    return (
        f'model={_h(row["model"])} object={_h(row["object"])} '
        f'tote={_h(row["tote_placement"])} outcome={_h(row["outcome"])} '
        f'S={_h(row["S"])}/N={_h(row["N"])} '
        f'expected\u2265{_h(row["expected_min"])} actual={_h(row["actual"])} '
        f'(cmd={_h(row["n_commanded"])}, acc={_h(row["n_accidental"])}, '
        f'orphan={_h(row["n_orphan_commanded"])}) '
        f'duration={_h(row["duration_s"])}s'
    )


def _episode_links(public_url: str) -> list[str]:
    """Per-episode header links. Only the public viewer is useful for
    review; raw https_dir / MP4 download links are skipped because the
    viewer is what the team actually uses (and the per-candidate links
    below already deep-link into it at the right timestamp)."""
    if not public_url:
        return []
    return [f'<a href="{html.escape(public_url)}" target="_blank">viewer</a>']


def _table_header() -> str:
    return (
        '<table><tr>'
        '<th class="kind">kind</th>'
        '<th>verdict</th>'
        '<th>ts_s</th>'
        '<th>paired_close_ts_s</th>'
        '<th>hold_ms</th>'
        '<th>disp_m</th>'
        '<th class="zone">zone</th>'
        '<th class="zone">paired_close_zone</th>'
        '<th>grip_before_150</th>'
        '<th>grip_after_150</th>'
        '<th>bal_before</th>'
        '<th>err\u00b1500</th>'
        '<th class="zone">reasons</th>'
        '<th class="links">links</th>'
        '</tr>'
    )


def _render_episode(row: dict[str, str], cands: list[dict[str, Any]]) -> str:
    src = row['source']
    rel = row['rel_path']
    public_url = row.get('public_url') or ''

    parts: list[str] = [f'<section class="episode" id="{html.escape(src)}-{html.escape(rel)}">']
    parts.append('<header>')
    parts.append(f'<span class="tag {row["status"]}">{row["status"].upper()}</span>')
    n_succ = sum(1 for c in cands if (c.get('features') or {}).get(FeatureName.VERDICT.slug) == Verdict.SUCCESS.slug)
    n_fail = sum(1 for c in cands if (c.get('features') or {}).get(FeatureName.VERDICT.slug) == Verdict.FAILURE.slug)
    n_unkn = sum(1 for c in cands if (c.get('features') or {}).get(FeatureName.VERDICT.slug) == Verdict.UNKNOWN.slug)
    parts.append(
        f'<span class="tag verdict success">{n_succ}\u00d7\u2713</span>'
        f'<span class="tag verdict failure">{n_fail}\u00d7\u2717</span>'
    )
    if n_unkn:
        parts.append(f'<span class="tag verdict unknown">{n_unkn}\u00d7?</span>')
    parts.append(f'<strong class="fixed">{html.escape(src)}/{html.escape(rel)}</strong>')
    parts.append(f'<span class="meta">{_episode_header_meta(row)}</span>')
    links = _episode_links(public_url)
    if links:
        parts.append('<span class="meta">' + ' \u00b7 '.join(links) + '</span>')
    parts.append('</header>')

    notes = (row.get('notes') or '').strip()
    if notes:
        parts.append(f'<div class="notes">notes: {html.escape(notes)}</div>')

    if not cands:
        parts.append('<p class="muted">No candidates emitted.</p>')
        parts.append('</section>')
        return '\n'.join(parts)

    parts.append(_table_header())
    for c in cands:
        parts.append(_candidate_row(c, public_url))
    parts.append('</table>')
    parts.append('</section>')
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Top-level page assembly.
# ---------------------------------------------------------------------------


def _render_summary(
    rows: list[dict[str, str]], cands_by_ep: dict[tuple[str, str], list[dict[str, Any]]]
) -> str:
    by_status: dict[str, int] = {}
    by_verdict_src: dict[tuple[str, str], int] = {}
    for r in rows:
        if Source.from_str(r['source']) is Source.HUMAN:
            continue
        by_status[r['status']] = by_status.get(r['status'], 0) + 1
        for c in cands_by_ep.get((r['source'], r['rel_path']), []):
            v = (c.get('features') or {}).get(FeatureName.VERDICT.slug) or 'invalid'
            by_verdict_src[(r['source'], v)] = by_verdict_src.get((r['source'], v), 0) + 1
    lines = ['<table class="summary-table">', '<tr><th>status</th><th class="num">episodes</th></tr>']
    for s in (RecallStatus.MISSING, RecallStatus.MATCH, RecallStatus.EXTRA):
        n = by_status.get(s.slug, 0)
        lines.append(f'<tr><td><span class="tag {s.slug}">{s.slug.upper()}</span></td><td class="num">{n}</td></tr>')
    lines.append('</table>')

    # Verdict counts per source.
    sources = sorted({src for (src, _) in by_verdict_src})
    if sources:
        lines.append('<table class="summary-table" style="margin-top:.5em">')
        lines.append(
            '<tr><th>source</th>'
            f'<th class="num">{Verdict.SUCCESS.slug}</th>'
            f'<th class="num">{Verdict.FAILURE.slug}</th>'
            f'<th class="num">{Verdict.UNKNOWN.slug}</th>'
            '</tr>'
        )
        for src in sources:
            ns = by_verdict_src.get((src, Verdict.SUCCESS.slug), 0)
            nf = by_verdict_src.get((src, Verdict.FAILURE.slug), 0)
            nu = by_verdict_src.get((src, Verdict.UNKNOWN.slug), 0)
            lines.append(
                f'<tr><td>{html.escape(src)}</td>'
                f'<td class="num">{ns}</td><td class="num">{nf}</td><td class="num">{nu}</td></tr>'
            )
        lines.append('</table>')
    return '\n'.join(lines)


def _render_intro() -> str:
    return (
        '<h1>Release-classifier stage-1 report</h1>'
        '<p class="meta">Recall-first detector output: every commanded release '
        '(<code>target_grip</code> 1\u21920 with <code>grip</code> non-zero) and '
        'every accidental drop (<code>grip</code> slump while <code>target_grip</code> '
        'stays locked). Episodes are grouped by recall status: '
        '<span class="tag missing">MISSING</span> (n &lt; expected) first, '
        'then <span class="tag match">MATCH</span>, then '
        '<span class="tag extra">EXTRA</span>. Click <span class="meta">view@&hellip;</span> '
        f'on any candidate row to open the public viewer paused {VIEWER_LEAD_IN_S:.1f}s before '
        'the candidate timestamp. Yellow boxes show <code>eval.notes</code> '
        '(inference only) where they exist.'
        '</p>'
    )


def _render(rows: list[dict[str, str]], cands_by_ep: dict[tuple[str, str], list[dict[str, Any]]]) -> str:
    parts: list[str] = []
    parts.append('<!doctype html>')
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append('<title>release_classifier stage-1 report</title>')
    parts.append(f'<style>{CSS}</style>')
    parts.append('</head><body>')
    parts.append(_render_intro())
    parts.append(_render_summary(rows, cands_by_ep))

    parts.append('<nav>')
    for s in (RecallStatus.MISSING, RecallStatus.MATCH, RecallStatus.EXTRA):
        parts.append(f'<a href="#sec-{s.slug}">{s.slug.upper()}</a>')
    parts.append('</nav>')

    for s in (RecallStatus.MISSING, RecallStatus.MATCH, RecallStatus.EXTRA):
        bucket = [r for r in rows if r['status'] == s.slug and Source.from_str(r['source']) is not Source.HUMAN]
        bucket.sort(key=lambda r: (r['source'], r['rel_path']))
        parts.append(f'<h2 id="sec-{s.slug}">{s.slug.upper()} ({len(bucket)})</h2>')
        if not bucket:
            parts.append('<p class="muted">(none)</p>')
            continue
        for r in bucket:
            parts.append(_render_episode(r, cands_by_ep.get((r['source'], r['rel_path']), [])))

    human_rows = [r for r in rows if Source.from_str(r['source']) is Source.HUMAN]
    if human_rows:
        parts.append(f'<h2 id="sec-human">HUMAN baselines &mdash; no robot signals ({len(human_rows)})</h2>')
        parts.append(
            '<p class="muted">These episodes record only the exterior camera. '
            'Stage 1 cannot detect releases here without vision.</p>'
        )
        for r in human_rows:
            parts.append(_render_episode(r, []))

    parts.append('</body></html>')
    return '\n'.join(parts)


def render(summary_path: Path, candidates_path: Path, out_path: Path) -> None:
    rows = _read_summary(summary_path)
    cands_by_ep = _read_candidates(candidates_path)
    html_str = _render(rows, cands_by_ep)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_str)
    print(f'Wrote {out_path}  ({len(rows)} episodes, {sum(len(v) for v in cands_by_ep.values())} candidates)')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=Path('out/release_classifier/episode_summary.csv'))
    parser.add_argument('--candidates', type=Path, default=Path('out/release_classifier/candidates.jsonl'))
    parser.add_argument('--out', type=Path, default=Path('out/release_classifier/report.html'))
    args = parser.parse_args()
    render(args.summary, args.candidates, args.out)


if __name__ == '__main__':
    main()
