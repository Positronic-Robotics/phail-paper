"""Render out/release_classifier/regressions_report.html — a focused report
on the subset of episodes that fail at least one cross-reference check
between the audit metadata and the stage-1/stage-2 detector output.

Structure (each level past the set header is collapsible via <details>):

  set (inference, training, ...)
    set-level stats (expected successes, detected candidates split by
    verdict, cross-check pass rates, manual-block hook)
    └── problematic episodes  (filterable, each as a <details> card)
        └── single episode
            ├── manual notes block (preserved across regenerations)
            ├── successes (table)
            ├── failures  (table)
            └── unknown / invalid (table, when present)

Cross-reference checks (an episode is "problematic" if ANY fails):
  - recall:        actual >= expected_min                    (status != MISSING)
  - success_count: detected commanded_release SUCCESS verdicts >= expected_min
  - no_orphan:     n_orphan_commanded == 0

Free-form sections:
  Each <!-- BEGIN MANUAL: <id> --> ... <!-- END MANUAL: <id> --> block is
  preserved verbatim across regenerations. Ids:
    manual-header                              (top of report)
    manual-set-<source_slug>                   (per set, e.g. manual-set-inference)
    manual-episode-<source>-<rel_path_slug>    (per problematic episode)

Inputs:
  - out/release_classifier/episode_summary.csv (from run_all)
  - out/release_classifier/candidates.jsonl    (from run_all)

Output:
  - out/release_classifier/regressions_report.html
    (existing manual blocks at the same path are merged in)

Usage:
    uv run python -m build.release_classifier.regressions_report
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import FeatureName, RecallStatus, Source, Verdict

# ---------------------------------------------------------------------------
# Style. Extends report.py CSS with details/summary, set-stats, check chips.
# ---------------------------------------------------------------------------

CSS = """
body { font: 13px/1.45 -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1280px; margin: 1.5em auto; color: #222; padding: 0 12px; }
h1 { margin: 0 0 .25em 0; }
h2 { margin: 1.5em 0 .25em 0; padding: .25em 0; border-bottom: 2px solid #888; }
h3 { margin: 1em 0 .25em 0; font-size: 14px; }
p.meta { color: #555; }
code { background: #f4f4f4; padding: 0 3px; border-radius: 2px; }

/* Tags reused from report.html */
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

/* Cross-check chips. */
.check { display: inline-block; padding: 1px 6px; border-radius: 10px;
         font-size: 11px; font-weight: 600; margin-right: 3px; border: 1px solid; }
.check.pass { color: #1e8449; background: #eafaf1; border-color: #abebc6; }
.check.fail { color: #922b21; background: #fdedec; border-color: #f5b7b1; }

.reason { display: inline-block; background: #eef; color: #335;
          border-radius: 3px; padding: 0 4px; margin-right: 3px;
          font-size: 11px; }
.reason.fail { background: #fee; color: #933; }

/* Tables */
table { border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 6px; }
th, td { padding: 3px 6px; border-bottom: 1px solid #eee; text-align: right; vertical-align: top; }
th:first-child, td:first-child { text-align: left; }
th { background: #eee; font-weight: 600; }
td.kind, th.kind, td.zone, th.zone { text-align: left; }
td.links, th.links { text-align: left; white-space: nowrap; }
td.links a { margin-right: .5em; }
.summary-table th { font-weight: 600; }
.summary-table td.num, .summary-table th.num { text-align: right; }

.warn { color: #c0392b; font-weight: 600; }
.muted { color: #888; }
.fixed { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }

/* Set-stats panel. */
.set-stats { display: flex; flex-wrap: wrap; gap: 2em; margin: .5em 0 1em 0;
             background: #fafafa; border: 1px solid #e6e6e6; border-radius: 4px;
             padding: 8px 14px; }
.set-stats table { width: auto; font-size: 12px; }
.set-stats th { background: transparent; text-align: left; font-weight: 500;
                color: #555; padding-right: 12px; }
.set-stats td { text-align: right; font-variant-numeric: tabular-nums; }
.set-stats .indent { padding-left: 1.5em; color: #777; }

/* Manual-content highlighting so authors can tell their content from generated. */
.manual { background: #fffaf0; border-left: 3px solid #f1c40f;
          padding: 6px 10px; margin: 6px 0; font-size: 12px; color: #333; }
.manual:empty { display: none; }
.manual .manual-label { display: block; color: #aa8800; font-size: 10px;
                        text-transform: uppercase; letter-spacing: 0.05em;
                        margin-bottom: 2px; }

/* Collapsible cards. */
details { margin: 4px 0; }
details > summary { cursor: pointer; user-select: none; padding: 4px 6px;
                    border-radius: 3px; outline: none; }
details > summary:hover { background: #f0f0f0; }
details > summary::-webkit-details-marker { color: #888; }

details.problematic-list { margin: .25em 0 1em 0; }
details.problematic-list > summary { font-weight: 600; font-size: 14px;
                                     padding: 6px 8px; background: #f0f3f7;
                                     border-radius: 3px; }

details.episode { border: 1px solid #ddd; border-radius: 4px; padding: 0;
                  margin: 6px 0; background: #fff; }
details.episode > summary { display: flex; flex-wrap: wrap; gap: .5em 1em;
                            align-items: baseline; padding: 8px 10px;
                            background: #fafafa; border-radius: 3px 3px 0 0; }
details.episode[open] > summary { border-bottom: 1px solid #eaeaea; }
details.episode .body { padding: 8px 12px 12px 12px; }

details.cand-group { border: 1px solid #eee; border-radius: 3px;
                     margin: 6px 0; padding: 0; background: #fff; }
details.cand-group > summary { padding: 4px 8px; background: #f7f7f7;
                               font-size: 12px; font-weight: 600; }
details.cand-group[open] > summary { border-bottom: 1px solid #eee; }
details.cand-group .group-body { padding: 0 8px 6px 8px; }

nav { font-size: 12px; margin: .5em 0; }
nav a { margin-right: 1em; }
"""


# ---------------------------------------------------------------------------
# Helpers.
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


def _candidate_verdict(c: dict[str, Any]) -> str:
    return (c.get('features') or {}).get(FeatureName.VERDICT.slug) or Verdict.INVALID.slug


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Cross-reference checks.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckSpec:
    """One cross-reference check between detector output and audit metadata.

    A check is `pass` when the detector and audit agree on the relevant
    quantity. An episode is "problematic" if any check fails.
    """

    slug: str
    label: str
    description: str


CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec(
        slug='recall',
        label='recall',
        description='actual ≥ expected_min (≥ S commanded releases for inference, ≥ N for training)',
    ),
    CheckSpec(
        slug='success_count',
        label='success_count',
        description='detected commanded_release verdict=SUCCESS count ≥ expected_min',
    ),
    CheckSpec(
        slug='no_orphan',
        label='no_orphan',
        description='n_orphan_commanded == 0 (every commanded release paired with a prior gripper close)',
    ),
)


@dataclass
class EpisodeView:
    """Per-episode joined view used by the renderer."""

    row: dict[str, str]
    cands: list[dict[str, Any]]
    successes: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    other: list[dict[str, Any]] = field(default_factory=list)
    n_commanded_success: int = 0
    n_commanded_failure: int = 0
    n_commanded_unknown: int = 0
    expected_min: int = 0
    actual: int = 0
    n_orphan: int = 0
    checks: dict[str, bool] = field(default_factory=dict)
    is_problematic: bool = False


def _build_view(row: dict[str, str], cands: list[dict[str, Any]]) -> EpisodeView:
    v = EpisodeView(row=row, cands=cands)
    for c in cands:
        verdict = _candidate_verdict(c)
        if verdict == Verdict.SUCCESS.slug:
            v.successes.append(c)
            if c.get('kind') == 'commanded_release':
                v.n_commanded_success += 1
        elif verdict == Verdict.FAILURE.slug:
            v.failures.append(c)
            if c.get('kind') == 'commanded_release':
                v.n_commanded_failure += 1
        else:
            v.other.append(c)
            if c.get('kind') == 'commanded_release':
                v.n_commanded_unknown += 1

    v.expected_min = _safe_int(row.get('expected_min'))
    v.actual = _safe_int(row.get('actual'))
    v.n_orphan = _safe_int(row.get('n_orphan_commanded'))
    v.checks = {
        'recall': v.actual >= v.expected_min,
        'success_count': v.n_commanded_success >= v.expected_min,
        'no_orphan': v.n_orphan == 0,
    }
    v.is_problematic = not all(v.checks.values())
    return v


# ---------------------------------------------------------------------------
# Manual-block preservation. <!-- BEGIN MANUAL: id --> ... <!-- END MANUAL: id -->
# blocks at the same id are copied from the previous file verbatim. Authors
# can edit content between the markers; the generator never touches it.
# ---------------------------------------------------------------------------


_MANUAL_RE = re.compile(
    r'<!--\s*BEGIN MANUAL:\s*([A-Za-z0-9_\-]+)\s*-->(?P<body>.*?)<!--\s*END MANUAL:\s*\1\s*-->', re.DOTALL
)


def _read_manual_blocks(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text()
    return {m.group(1): m.group('body') for m in _MANUAL_RE.finditer(text)}


def _emit_manual_block(blocks: dict[str, str], block_id: str, hint: str = '') -> str:
    """Emit a <!-- BEGIN/END MANUAL --> region that round-trips across regens.

    On first emission (block_id not in `blocks`) the body is a small comment
    telling the author this region is editable. On regen, the previous body
    is preserved verbatim — including any HTML the author wrote.
    """
    if block_id in blocks:
        body = blocks[block_id]
    else:
        body = f'\n<!-- {hint or "Edit freely; preserved across regenerations."} -->\n'
    return f'<!-- BEGIN MANUAL: {block_id} -->{body}<!-- END MANUAL: {block_id} -->'


# ---------------------------------------------------------------------------
# Per-candidate row.
# ---------------------------------------------------------------------------


VIEWER_LEAD_IN_S: float = 1.5


def _viewer_url(public_url: str, ts_s: float) -> str | None:
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
        f'<span class="reason{(" fail" if is_fail else "")}">{html.escape(str(r))}</span>' for r in reasons
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
        '<th>err±500</th>'
        '<th class="zone">reasons</th>'
        '<th class="links">links</th>'
        '</tr>'
    )


# ---------------------------------------------------------------------------
# Per-episode card.
# ---------------------------------------------------------------------------


def _check_chips(checks: dict[str, bool]) -> str:
    parts: list[str] = []
    for spec in CHECKS:
        ok = checks.get(spec.slug, True)
        cls = 'pass' if ok else 'fail'
        sym = '✓' if ok else '✗'
        parts.append(
            f'<span class="check {cls}" title="{html.escape(spec.description)}">{sym} {html.escape(spec.label)}</span>'
        )
    return ''.join(parts)


def _cand_group(group_id: str, label: str, cands: list[dict[str, Any]], public_url: str, default_open: bool) -> str:
    if not cands:
        return ''
    open_attr = ' open' if default_open else ''
    rows = '\n'.join(_candidate_row(c, public_url) for c in cands)
    return (
        f'<details class="cand-group" id="{html.escape(group_id)}"{open_attr}>'
        f'<summary>{html.escape(label)} ({len(cands)})</summary>'
        f'<div class="group-body">{_table_header()}{rows}</table></div>'
        '</details>'
    )


def _episode_id(row: dict[str, str]) -> str:
    return f'ep-{row["source"]}-{row["rel_path"].replace("/", "-")}'


def _episode_summary(row: dict[str, str], view: EpisodeView) -> str:
    public_url = row.get('public_url') or ''
    cands_summary = (
        f'<span class="tag verdict success">{view.n_commanded_success}×✓</span>'
        f'<span class="tag verdict failure">{view.n_commanded_failure}×✗</span>'
    )
    if view.n_commanded_unknown:
        cands_summary += f'<span class="tag verdict unknown">{view.n_commanded_unknown}×?</span>'
    meta = (
        f'model={_h(row["model"])} object={_h(row["object"])} '
        f'tote={_h(row["tote_placement"])} outcome={_h(row["outcome"])} '
        f'S={_h(row["S"])}/N={_h(row["N"])} '
        f'expected≥{_h(row["expected_min"])} actual={_h(row["actual"])} '
        f'(cmd={_h(row["n_commanded"])}, acc={_h(row["n_accidental"])}, '
        f'orphan={_h(row["n_orphan_commanded"])}) '
        f'duration={_h(row["duration_s"])}s'
    )
    link = f'<a href="{html.escape(public_url)}" target="_blank">phail.ai</a>' if public_url else ''
    return (
        '<summary>'
        f'<span class="tag {row["status"]}">{row["status"].upper()}</span>'
        f'{cands_summary}'
        f'<strong class="fixed">{html.escape(row["source"])}/{html.escape(row["rel_path"])}</strong>'
        f'<span class="check-chips">{_check_chips(view.checks)}</span>'
        f'<span class="meta">{meta}</span>'
        f'<span class="meta">{link}</span>'
        '</summary>'
    )


def _audit_notes_block(row: dict[str, str]) -> str:
    notes = (row.get('notes') or '').strip()
    if not notes:
        return ''
    return f'<div class="manual"><span class="manual-label">audit eval.notes</span>{html.escape(notes)}</div>'


def _checks_detail(view: EpisodeView) -> str:
    rows: list[str] = []
    for spec in CHECKS:
        ok = view.checks.get(spec.slug, True)
        cls = 'pass' if ok else 'fail'
        sym = '✓' if ok else '✗'
        rows.append(
            f'<tr><td><span class="check {cls}">{sym} {html.escape(spec.label)}</span></td>'
            f'<td class="muted">{html.escape(spec.description)}</td></tr>'
        )
    rows_html = '\n'.join(rows)
    return (
        '<details class="cand-group">'
        '<summary>Cross-reference checks</summary>'
        f'<div class="group-body"><table>{rows_html}</table></div>'
        '</details>'
    )


def _render_episode(view: EpisodeView, manual_blocks: dict[str, str]) -> str:
    row = view.row
    public_url = row.get('public_url') or ''
    ep_id = _episode_id(row)
    manual_id = f'manual-episode-{row["source"]}-{row["rel_path"].replace("/", "-")}'
    parts: list[str] = [f'<details class="episode" id="{html.escape(ep_id)}">']
    parts.append(_episode_summary(row, view))
    parts.append('<div class="body">')
    parts.append(_audit_notes_block(row))
    parts.append(
        '<div class="manual"><span class="manual-label">manual notes</span>'
        + _emit_manual_block(
            manual_blocks, manual_id, 'Hand-written notes for this episode go here. Preserved across regenerations.'
        )
        + '</div>'
    )
    parts.append(_checks_detail(view))
    parts.append(_cand_group(f'{ep_id}-success', 'Successes', view.successes, public_url, default_open=True))
    parts.append(_cand_group(f'{ep_id}-failure', 'Failures', view.failures, public_url, default_open=False))
    parts.append(_cand_group(f'{ep_id}-other', 'Unknown / invalid', view.other, public_url, default_open=False))
    if not view.cands:
        parts.append('<p class="muted">No candidates emitted.</p>')
    parts.append('</div></details>')
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Per-set stats and section.
# ---------------------------------------------------------------------------


@dataclass
class SetStats:
    n_total: int = 0
    n_problematic: int = 0
    sum_expected: int = 0
    sum_actual: int = 0
    sum_orphan: int = 0
    n_success: int = 0
    n_failure: int = 0
    n_unknown: int = 0
    n_invalid: int = 0
    check_pass: dict[str, int] = field(default_factory=lambda: {c.slug: 0 for c in CHECKS})

    def add(self, view: EpisodeView) -> None:
        self.n_total += 1
        if view.is_problematic:
            self.n_problematic += 1
        self.sum_expected += view.expected_min
        self.sum_actual += view.actual
        self.sum_orphan += view.n_orphan
        for c in view.cands:
            v = _candidate_verdict(c)
            if v == Verdict.SUCCESS.slug:
                self.n_success += 1
            elif v == Verdict.FAILURE.slug:
                self.n_failure += 1
            elif v == Verdict.UNKNOWN.slug:
                self.n_unknown += 1
            else:
                self.n_invalid += 1
        for spec in CHECKS:
            if view.checks.get(spec.slug, True):
                self.check_pass[spec.slug] += 1


def _set_stats_panel(stats: SetStats) -> str:
    n = stats.n_total
    rows = [
        ('total episodes (non-human)', f'{stats.n_total}'),
        ('problematic episodes', f'{stats.n_problematic}'),
        ('Σ expected successes', f'{stats.sum_expected}'),
        ('Σ actual detected candidates', f'{stats.sum_actual}'),
        ('<span class="indent">success</span>', f'{stats.n_success}'),
        ('<span class="indent">failure</span>', f'{stats.n_failure}'),
        ('<span class="indent">unknown</span>', f'{stats.n_unknown}'),
        ('<span class="indent">invalid</span>', f'{stats.n_invalid}'),
        ('Σ orphan commanded releases', f'{stats.sum_orphan}'),
    ]
    rows_html = '\n'.join(f'<tr><th>{label}</th><td>{value}</td></tr>' for label, value in rows)
    check_rows = []
    for spec in CHECKS:
        passed = stats.check_pass.get(spec.slug, 0)
        pct = f'{(100.0 * passed / n):.1f}%' if n else '–'
        check_rows.append(
            f'<tr><th><span class="indent">{html.escape(spec.label)}</span></th><td>{passed} / {n} ({pct})</td></tr>'
        )
    check_rows_html = '\n'.join(check_rows)
    return (
        '<div class="set-stats">'
        f'<table><tbody>{rows_html}</tbody></table>'
        '<table><thead><tr><th>cross-check</th><th>pass</th></tr></thead>'
        f'<tbody>{check_rows_html}</tbody></table>'
        '</div>'
    )


def _render_set(src_slug: str, views: list[EpisodeView], manual_blocks: dict[str, str]) -> str:
    stats = SetStats()
    for v in views:
        stats.add(v)
    problematic = [v for v in views if v.is_problematic]
    problematic.sort(key=lambda v: v.row['rel_path'])

    parts: list[str] = []
    parts.append(
        f'<h2 id="sec-{html.escape(src_slug)}">{html.escape(src_slug)} '
        f'— {stats.n_problematic} problematic / {stats.n_total} total</h2>'
    )
    parts.append(_set_stats_panel(stats))
    parts.append(
        '<div class="manual"><span class="manual-label">set notes</span>'
        + _emit_manual_block(
            manual_blocks,
            f'manual-set-{src_slug}',
            f'Hand-written notes about the {src_slug} set. Preserved across regenerations.',
        )
        + '</div>'
    )

    if not problematic:
        parts.append('<p class="muted">No problematic episodes for this set.</p>')
        return '\n'.join(parts)

    parts.append(f'<details class="problematic-list" open><summary>Problematic episodes ({len(problematic)})</summary>')
    for v in problematic:
        parts.append(_render_episode(v, manual_blocks))
    parts.append('</details>')
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Top-level page.
# ---------------------------------------------------------------------------


def _render_intro(manual_blocks: dict[str, str]) -> str:
    parts: list[str] = []
    parts.append('<h1>Release classifier — potentially problematic episodes</h1>')
    parts.append(
        '<p class="meta">Episodes that fail at least one cross-reference check '
        'between the audit metadata and the stage-1/stage-2 detector output. '
        'Use this view to spot detector regressions, audit metadata errors, '
        'and edge cases worth a manual look. The full per-episode report '
        '(every episode, not just problematic ones) lives in '
        '<code>report.html</code>.'
        '</p>'
    )
    parts.append(
        '<p class="meta"><strong>Cross-reference checks</strong>: '
        + ' &nbsp;·&nbsp; '.join(f'<code>{html.escape(c.label)}</code> {html.escape(c.description)}' for c in CHECKS)
        + '. An episode is flagged problematic if any check fails.</p>'
    )
    parts.append(
        '<p class="meta">Click <span class="meta">view@…</span> on any '
        f'candidate to open the phail.ai viewer paused {VIEWER_LEAD_IN_S:.1f}s '
        'before the candidate timestamp. Yellow boxes are editable manual '
        'sections — content between <code>&lt;!-- BEGIN MANUAL: id --&gt;</code> '
        'and <code>&lt;!-- END MANUAL: id --&gt;</code> is preserved across '
        'regenerations. Edit the file directly to add notes.</p>'
    )
    parts.append(
        '<div class="manual"><span class="manual-label">report-level notes</span>'
        + _emit_manual_block(
            manual_blocks,
            'manual-header',
            'Top-level notes (methodology, hypotheses, takeaways). Preserved across regenerations.',
        )
        + '</div>'
    )
    return '\n'.join(parts)


def _render_top_summary(views_by_src: dict[str, list[EpisodeView]]) -> str:
    rows: list[str] = []
    rows.append(
        '<tr><th>set</th><th class="num">total</th><th class="num">problematic</th>'
        '<th class="num">Σ expected</th><th class="num">Σ actual</th>'
        '<th class="num">success</th><th class="num">failure</th>'
        '<th class="num">orphan</th></tr>'
    )
    for src in sorted(views_by_src):
        stats = SetStats()
        for v in views_by_src[src]:
            stats.add(v)
        rows.append(
            f'<tr><td>{html.escape(src)}</td>'
            f'<td class="num">{stats.n_total}</td>'
            f'<td class="num">{stats.n_problematic}</td>'
            f'<td class="num">{stats.sum_expected}</td>'
            f'<td class="num">{stats.sum_actual}</td>'
            f'<td class="num">{stats.n_success}</td>'
            f'<td class="num">{stats.n_failure}</td>'
            f'<td class="num">{stats.sum_orphan}</td>'
            '</tr>'
        )
    return '<table class="summary-table">' + '\n'.join(rows) + '</table>'


def _render_nav(views_by_src: dict[str, list[EpisodeView]]) -> str:
    items = []
    for src in sorted(views_by_src):
        n_prob = sum(1 for v in views_by_src[src] if v.is_problematic)
        items.append(f'<a href="#sec-{html.escape(src)}">{html.escape(src)} ({n_prob})</a>')
    return '<nav>' + ''.join(items) + '</nav>'


def _render(
    rows: list[dict[str, str]], cands_by_ep: dict[tuple[str, str], list[dict[str, Any]]], manual_blocks: dict[str, str]
) -> str:
    views_by_src: dict[str, list[EpisodeView]] = {}
    for r in rows:
        if Source.from_str(r['source']) is Source.HUMAN:
            continue  # No robot signals; out of scope here.
        if RecallStatus.from_str(r['status']) is RecallStatus.INVALID:
            continue
        cands = cands_by_ep.get((r['source'], r['rel_path']), [])
        view = _build_view(r, cands)
        views_by_src.setdefault(r['source'], []).append(view)

    parts: list[str] = []
    parts.append('<!doctype html>')
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append('<title>release_classifier — potentially problematic episodes</title>')
    parts.append(f'<style>{CSS}</style>')
    parts.append('</head><body>')
    parts.append(_render_intro(manual_blocks))
    parts.append(_render_top_summary(views_by_src))
    parts.append(_render_nav(views_by_src))
    for src in sorted(views_by_src):
        parts.append(_render_set(src, views_by_src[src], manual_blocks))
    parts.append('</body></html>')
    return '\n'.join(parts)


def render(summary_path: Path, candidates_path: Path, out_path: Path) -> None:
    rows = _read_summary(summary_path)
    cands_by_ep = _read_candidates(candidates_path)
    manual_blocks = _read_manual_blocks(out_path)
    html_str = _render(rows, cands_by_ep, manual_blocks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_str)
    n_problematic = sum(
        1
        for r in rows
        if Source.from_str(r['source']) is not Source.HUMAN
        and RecallStatus.from_str(r['status']) is not RecallStatus.INVALID
        and _build_view(r, cands_by_ep.get((r['source'], r['rel_path']), [])).is_problematic
    )
    print(
        f'Wrote {out_path}  ({len(rows)} episodes scanned, '
        f'{n_problematic} flagged problematic, '
        f'{len(manual_blocks)} manual blocks preserved)'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=Path('out/release_classifier/episode_summary.csv'))
    parser.add_argument('--candidates', type=Path, default=Path('out/release_classifier/candidates.jsonl'))
    parser.add_argument('--out', type=Path, default=Path('out/release_classifier/regressions_report.html'))
    args = parser.parse_args()
    render(args.summary, args.candidates, args.out)


if __name__ == '__main__':
    main()
