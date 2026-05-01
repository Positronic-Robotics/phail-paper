"""Manual markup UI for D9 episode logs.

Three flows on the same UI:

1. **Free scrub** (legacy): scrub the wrist + exterior cameras, hit `A` to
   add a place at the cursor, `[/]` to bump items_in_source_at_end. Useful
   when you want to mark places without a stage-1 candidate list.

2. **Candidate review** (preferred when stage 1+2 has run): pass
   `--candidates out/release_classifier/candidates.jsonl`. The UI walks
   each stage-1 candidate in order: it seeks 0.5 s *before* the candidate's
   timestamp, plays through, auto-pauses 0.5 s *after* it, and prompts you
   for two integers — `n_successful` (items the gripper actually placed in
   the target bin) and `n_out_of_bins` (items ending up outside both bins).
   Each field is pre-populated from the stage-2 verdict (1 for SUCCESS or
   FAILURE, respectively); typing a digit replaces the active field, Tab
   toggles which field is active, Enter saves and advances. At the end of
   an episode the `,/.` keys are highlighted so you can switch episodes.

3. **Free scrub with candidate overlay** (calibration / ground-truth pass):
   pass both `--candidates ...` AND `--free_scrub`. Free-scrub keys are
   active (A / X / D / [ / ]); candidates render as non-actionable
   light-blue ticks on the timeline. Lets the annotator see where stage-1
   fired without being constrained to its event set, which is the right
   shape for a calibration set: manual is the ground truth, candidates are
   reference. Stage-1 misses are visible immediately as gaps.

Annotations are saved to `{out_dir}/{rel_id}.json` (auto-save on every
edit). The source dataset is never modified — pos3 is read-only.

Episode list is provided via either:
  --debug_set      build/autopilot/debug_set.json   (curated)
  --episode_root   <dir>                            (walk a tree)

Both s3:// and local paths work via pos3.

Usage:
    uv run python -m build.markup.ui --debug_set build/autopilot/debug_set.json
    uv run python -m build.markup.ui --debug_set <set> --candidates out/release_classifier/candidates.jsonl
    uv run python -m build.markup.ui --episode_root ~/.cache/.../inference --candidates ...
"""

from __future__ import annotations

import faulthandler
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Surface a Python+C stack trace if a future PyAV/dearpygui crash slips
# through the preload guard, instead of exiting silently.
faulthandler.enable()

import configuronic as cfn
import dearpygui.dearpygui as dpg
import numpy as np
import pos3

# Public-S3 profile so pos3 syncs with the phail dataset bucket without
# needing AWS credentials. Mirrors `build/data_audit/build_index.py`.
PUBLIC_PROFILE = pos3.Profile(
    local_name='positronic-public', endpoint='https://storage.eu-north1.nebius.cloud', public=True, region='eu-north1'
)

from positronic.dataset.local_dataset import DiskEpisode
from positronic.dataset.signal import Signal

from . import io as markup_io

log = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = 's3://positronic-public/datasets/phail/v1.0/inference'
DEFAULT_OUT_DIR = str(Path(__file__).resolve().parent / 'annotations')

SEEK_NUDGE_S = 0.5
SEEK_SHIFT_S = 1.0
SEEK_CTRL_S = 5.0
SEEK_BOTH_S = 10.0
DISPLAY_HEIGHT = 380

# Candidate review timing — see module docstring.
CAND_LEAD_IN_S = 0.5
CAND_LEAD_OUT_S = 0.5

# Verdict slugs as written by build/release_classifier/classify.py.
VERDICT_SUCCESS = 'success'
VERDICT_FAILURE = 'failure'
VERDICT_NEUTRAL = 'neutral'  # release back into source bin — auto-skipped from review

ACTIVE_COLOR = (255, 220, 0, 255)
INACTIVE_COLOR = (220, 220, 220, 255)
DIM_COLOR = (140, 140, 140, 255)
HINT_COLOR = (120, 200, 255, 255)
HINT_HIGHLIGHT_COLOR = (255, 100, 100, 255)


@dataclass
class EpisodeRef:
    abs_path: Path
    rel_id: str  # e.g. "000000000000/000000000293"
    source: str  # 'inference' / 'training' / 'human' — needed because rel_ids collide across sources


@dataclass
class Candidate:
    """One stage-1 release candidate to review."""

    ts_s: float
    verdict: str  # 'success' / 'failure' / 'unknown' / 'invalid'
    kind: str  # 'commanded_release' / 'accidental_drop' / ...
    reasons: list[str]


# ---------------------------------------------------------------------------
# Candidate loading.
# ---------------------------------------------------------------------------


def _load_candidates_by_episode(path: Path) -> dict[tuple[str, str], list[Candidate]]:
    """Group candidates from a stage-2 candidates.jsonl by `(source, rel_path)`.

    Keyed by the (source, rel_path) tuple — *not* rel_path alone — because
    inference/000000000000/000000000028 and training/000000000000/000000000028
    are different episodes that happen to share a numerical id; merging
    them in the UI shows wrong timestamps for whichever video the user is
    actually watching.
    """
    out: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            feat = rec.get('features') or {}
            key = (str(rec.get('source') or ''), str(rec.get('rel_path') or ''))
            out[key].append(
                Candidate(
                    ts_s=float(rec['ts_s']),
                    verdict=str(feat.get('verdict') or 'invalid'),
                    kind=str(rec.get('kind') or ''),
                    reasons=list(feat.get('verdict_reasons') or []),
                )
            )
    for cands in out.values():
        cands.sort(key=lambda c: c.ts_s)
    return dict(out)


def _bootstrap_annotations(
    candidates: list[Candidate], existing: list[markup_io.CandidateAnnotation]
) -> list[markup_io.CandidateAnnotation]:
    """One annotation per stage-1 candidate. Pre-populated from the verdict
    when no prior annotation exists; existing user edits win on reload."""
    by_ts = {round(a.ts_s, 3): a for a in existing}
    out: list[markup_io.CandidateAnnotation] = []
    for c in candidates:
        key = round(c.ts_s, 3)
        if key in by_ts:
            out.append(by_ts[key])
            continue
        out.append(
            markup_io.CandidateAnnotation(
                ts_s=c.ts_s,
                n_successful=1 if c.verdict == VERDICT_SUCCESS else 0,
                n_out_of_bins=1 if c.verdict == VERDICT_FAILURE else 0,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Reviewability helpers: NEUTRAL candidates (releases back into the source
# bin — fumbles / re-grips) auto-resolve to 0/0 and aren't actionable, so
# the review loop skips them entirely. The forward/back navigation jumps
# directly to the nearest reviewable candidate.
# ---------------------------------------------------------------------------


def _is_reviewable(c: Candidate) -> bool:
    return c.verdict != VERDICT_NEUTRAL


def _next_reviewable(view: EpisodeView, start_idx: int) -> int | None:
    for i in range(start_idx, len(view.candidates)):
        if _is_reviewable(view.candidates[i]):
            return i
    return None


def _prev_reviewable(view: EpisodeView, start_idx: int) -> int | None:
    for i in range(start_idx, -1, -1):
        if _is_reviewable(view.candidates[i]):
            return i
    return None


def _first_reviewable(view: EpisodeView) -> int | None:
    return _next_reviewable(view, 0) if view.candidates else None


# ---------------------------------------------------------------------------
# Episode + App state.
# ---------------------------------------------------------------------------


class EpisodeView:
    """Per-episode mutable state."""

    def __init__(
        self,
        ref: EpisodeRef,
        out_dir: Path,
        candidates: list[Candidate] | None,
        free_scrub: bool = False,
    ) -> None:
        self.ref = ref
        self.markup_path = markup_io.markup_path(out_dir, ref.rel_id, ref.source)
        self.ep = DiskEpisode(ref.abs_path)
        self.start_ts = int(self.ep.start_ts)
        self.last_ts = int(self.ep.last_ts)
        self.duration_s = self.ep.duration_ns / 1e9
        self.video_wrist = self.ep['image.wrist']
        self.video_ext = self.ep['image.exterior']
        legacy_ts, items_in_source, prior_annotations, prior_drops, reviewed = markup_io.load_full(
            self.markup_path
        )
        self.items_in_source = items_in_source
        self.candidates: list[Candidate] = candidates or []
        # `free_scrub=True` forces the free-scrub workflow even when candidates
        # are loaded — they become a non-actionable timeline overlay so the
        # annotator can see where stage-1 fired while marking placements
        # manually. Useful for the calibration set, where manual is the
        # ground truth (don't bias toward what stage-1 already found).
        self.free_scrub = free_scrub
        if self.candidates and not self.free_scrub:
            self.cand_annotations = _bootstrap_annotations(self.candidates, prior_annotations)
            self.place_timestamps_s = markup_io.derive_place_timestamps_from_annotations(self.cand_annotations)
        else:
            self.cand_annotations = list(prior_annotations)
            self.place_timestamps_s = list(legacy_ts)
        self.drop_timestamps_s: list[float] = list(prior_drops)
        self.reviewed: bool = reviewed
        self.selected_idx: int | None = 0 if self.place_timestamps_s else None
        self.cursor_s = 0.0
        self.playing = False
        # Candidate review state.
        self.cand_idx: int = 0 if self.candidates else -1
        self.active_field: str = 'n_successful'  # or 'n_out_of_bins'
        self.candidate_paused: bool = False  # set True at the auto-pause point

        self.meta = {}
        for key in self.ep.keys():
            val = self.ep[key]
            if not isinstance(val, Signal):
                self.meta[key] = val

    @property
    def total_items(self) -> int:
        return int(self.meta.get('eval.total_items') or 0)

    def frame_at_cursor(self, video: Signal) -> np.ndarray:
        ts = self.start_ts + int(self.cursor_s * 1e9)
        ts = max(self.start_ts, min(self.last_ts, ts))
        frame, _ = video.time[ts]
        return frame

    def has_candidate_workflow(self) -> bool:
        return bool(self.candidates) and not self.free_scrub

    def has_candidate_overlay(self) -> bool:
        """Candidates loaded for visualization, regardless of workflow."""
        return bool(self.candidates)

    def at_episode_end(self) -> bool:
        """In candidate-review mode, True iff every candidate has been visited."""
        return self.has_candidate_workflow() and self.cand_idx >= len(self.candidates)

    def active_candidate(self) -> Candidate | None:
        if not self.has_candidate_workflow():
            return None
        if 0 <= self.cand_idx < len(self.candidates):
            return self.candidates[self.cand_idx]
        return None

    def active_annotation(self) -> markup_io.CandidateAnnotation | None:
        if not self.has_candidate_workflow():
            return None
        if 0 <= self.cand_idx < len(self.cand_annotations):
            return self.cand_annotations[self.cand_idx]
        return None


class App:
    def __init__(
        self,
        episodes: list[EpisodeRef],
        out_dir: Path,
        candidates_by_ep: dict[tuple[str, str], list[Candidate]] | None,
        free_scrub: bool = False,
    ):
        if not episodes:
            raise ValueError('no episodes provided')
        self.episodes = episodes
        self.out_dir = out_dir
        self.candidates_by_ep = candidates_by_ep or {}
        self.free_scrub = free_scrub
        # We *cache* EpisodeView instances rather than recreate them on every
        # navigation. Tearing down a DiskEpisode (and the libav containers it
        # owns) while a new one's first frame is being decoded triggers a
        # libmalloc abort inside av_fast_realloc — saw a SIGABRT crash report
        # on the first `.` press. Holding all visited views alive sidesteps
        # the teardown race entirely. Memory cost per view is small (just
        # parquet metadata + lazily-decoded numpy frames).
        self._view_cache: dict[int, EpisodeView] = {}
        # Pre-warm every EpisodeView (open libav containers + decode one
        # frame per video stream) at startup, in a single controlled
        # sequence. Without this, the second time the user presses `.`
        # to switch episodes we get a SIGABRT inside av_fast_realloc:
        # PyAV / libav has a known race where opening a fresh container
        # while a frame from a different container is being decoded
        # corrupts a malloc bucket. Doing all opens up front (no decodes
        # interleaved with key-press handlers) sidesteps it. Cost is a
        # one-time preload at startup (~1 s × N episodes); during
        # interactive use, switch_episode is just a Python-object swap.
        log.info('preloading %d episodes...', len(episodes))
        for i in range(len(episodes)):
            view = self._make_view(i)
            try:
                view.frame_at_cursor(view.video_wrist)
                view.frame_at_cursor(view.video_ext)
            except Exception as exc:  # missing video file, decode error, etc.
                log.warning('  ep %d (%s) preload failed: %r', i, episodes[i].rel_id, exc)
            if (i + 1) % 5 == 0 or i == len(episodes) - 1:
                log.info('  preloaded %d/%d', i + 1, len(episodes))
        log.info('preload complete; switch_episode is now decode-free')
        self.idx = 0
        self.view: EpisodeView = self._view_cache[0]
        self.tex_wrist_shape: tuple[int, int] | None = None
        self.tex_ext_shape: tuple[int, int] | None = None
        self.tex_wrist: np.ndarray | None = None
        self.tex_ext: np.ndarray | None = None
        self.last_save_t: float | None = None
        self.last_render_t = time.monotonic()

    def _make_view(self, idx: int) -> EpisodeView:
        if idx in self._view_cache:
            return self._view_cache[idx]
        ref = self.episodes[idx]
        cands = self.candidates_by_ep.get((ref.source, ref.rel_id)) if self.candidates_by_ep else None
        view = EpisodeView(ref, self.out_dir, cands, free_scrub=self.free_scrub)
        self._view_cache[idx] = view
        return view

    # ---- episode lifecycle ----

    def switch_episode(self, new_idx: int) -> None:
        new_idx = max(0, min(len(self.episodes) - 1, new_idx))
        if new_idx == self.idx:
            return
        self.idx = new_idx
        self.view = self._make_view(new_idx)  # cached: see App.__init__ note
        # Do NOT decode here. switch_episode runs on dpg's input thread; the
        # render loop decodes on its own thread inside refresh_frames. Two
        # simultaneous PyAV decodes (even on different containers) corrupt
        # libav's malloc state and abort the process — saw a SIGABRT on `.`
        # with both threads stuck in seek_if_needed / __next__. The render
        # loop's refresh_frames already rebuilds textures on shape mismatch,
        # so leave all decode + texture work to it.
        self.last_save_t = None
        if self.view.has_candidate_workflow():
            self.jump_to_candidate(_first_reviewable(self.view) or 0, auto_play=False)

    def _ensure_textures(self) -> None:
        wrist = self.view.frame_at_cursor(self.view.video_wrist)
        ext = self.view.frame_at_cursor(self.view.video_ext)
        wrist_shape = wrist.shape[:2]
        ext_shape = ext.shape[:2]

        if wrist_shape != self.tex_wrist_shape:
            if dpg.does_item_exist('tex_wrist'):
                dpg.delete_item('tex_wrist')
            self.tex_wrist = np.ones((*wrist_shape, 4), dtype=np.float32)
            with dpg.texture_registry():
                dpg.add_raw_texture(
                    width=wrist_shape[1],
                    height=wrist_shape[0],
                    tag='tex_wrist',
                    format=dpg.mvFormat_Float_rgba,
                    default_value=self.tex_wrist,
                )
            self.tex_wrist_shape = wrist_shape
            if dpg.does_item_exist('img_wrist'):
                w, h = _display_size(wrist_shape)
                dpg.configure_item('img_wrist', texture_tag='tex_wrist', width=w, height=h)

        if ext_shape != self.tex_ext_shape:
            if dpg.does_item_exist('tex_ext'):
                dpg.delete_item('tex_ext')
            self.tex_ext = np.ones((*ext_shape, 4), dtype=np.float32)
            with dpg.texture_registry():
                dpg.add_raw_texture(
                    width=ext_shape[1],
                    height=ext_shape[0],
                    tag='tex_ext',
                    format=dpg.mvFormat_Float_rgba,
                    default_value=self.tex_ext,
                )
            self.tex_ext_shape = ext_shape
            if dpg.does_item_exist('img_ext'):
                w, h = _display_size(ext_shape)
                dpg.configure_item('img_ext', texture_tag='tex_ext', width=w, height=h)

    def refresh_frames(self) -> None:
        wrist = self.view.frame_at_cursor(self.view.video_wrist)
        ext = self.view.frame_at_cursor(self.view.video_ext)
        if wrist.shape[:2] != self.tex_wrist_shape or ext.shape[:2] != self.tex_ext_shape:
            self._ensure_textures()
            wrist = self.view.frame_at_cursor(self.view.video_wrist)
            ext = self.view.frame_at_cursor(self.view.video_ext)
        self.tex_wrist[:, :, :3] = wrist.astype(np.float32) / 255.0
        self.tex_ext[:, :, :3] = ext.astype(np.float32) / 255.0

    # ---- mutations (auto-save) ----

    def _persist(self) -> None:
        if self.view.has_candidate_workflow():
            # Source-of-truth = candidate annotations; place_timestamps_s is derived.
            self.view.place_timestamps_s = markup_io.derive_place_timestamps_from_annotations(
                self.view.cand_annotations
            )
            markup_io.save_markup(
                self.view.markup_path,
                self.view.place_timestamps_s,
                self.view.items_in_source,
                candidate_annotations=self.view.cand_annotations,
                drop_timestamps_s=self.view.drop_timestamps_s,
                reviewed=self.view.reviewed,
            )
        else:
            markup_io.save_markup(
                self.view.markup_path,
                self.view.place_timestamps_s,
                self.view.items_in_source,
                drop_timestamps_s=self.view.drop_timestamps_s,
                reviewed=self.view.reviewed,
            )
        self.last_save_t = time.monotonic()

    # ---- additive flags (work in both flows) ----

    def toggle_reviewed(self) -> None:
        self.view.reviewed = not self.view.reviewed
        self._persist()

    # ---- free-scrub drop events (no candidate-mode equivalent needed —
    # candidate mode captures drops as `n_out_of_bins` per candidate) ----

    def add_drop(self) -> None:
        if self.view.has_candidate_workflow():
            return  # candidate mode uses n_out_of_bins; drop_timestamps_s is free-scrub-only
        t = round(self.view.cursor_s, 3)
        self.view.drop_timestamps_s.append(t)
        self.view.drop_timestamps_s.sort()
        self._persist()

    def delete_nearest_drop(self) -> None:
        if self.view.has_candidate_workflow() or not self.view.drop_timestamps_s:
            return
        t = self.view.cursor_s
        nearest = min(range(len(self.view.drop_timestamps_s)),
                      key=lambda i: abs(self.view.drop_timestamps_s[i] - t))
        del self.view.drop_timestamps_s[nearest]
        self._persist()

    # ---- legacy free-scrub mutations (still available without candidates) ----

    def add_place(self) -> None:
        if self.view.has_candidate_workflow():
            return  # candidate mode owns the place list
        self.view.place_timestamps_s.append(round(self.view.cursor_s, 3))
        self.view.place_timestamps_s.sort()
        self.view.selected_idx = self.view.place_timestamps_s.index(round(self.view.cursor_s, 3))
        self._persist()

    def delete_selected_place(self) -> None:
        if self.view.has_candidate_workflow():
            return
        idx = self.view.selected_idx
        if idx is None or not (0 <= idx < len(self.view.place_timestamps_s)):
            return
        del self.view.place_timestamps_s[idx]
        if not self.view.place_timestamps_s:
            self.view.selected_idx = None
        else:
            self.view.selected_idx = min(idx, len(self.view.place_timestamps_s) - 1)
        self._persist()

    def select_place(self, delta: int) -> None:
        if not self.view.place_timestamps_s:
            return
        if self.view.selected_idx is None:
            self.view.selected_idx = 0
        else:
            self.view.selected_idx = (self.view.selected_idx + delta) % len(self.view.place_timestamps_s)
        self.view.cursor_s = self.view.place_timestamps_s[self.view.selected_idx]
        self.view.playing = False

    def adjust_source(self, delta: int) -> None:
        new_val = self.view.items_in_source + delta
        new_val = max(0, min(self.view.total_items, new_val))
        if new_val == self.view.items_in_source:
            return
        self.view.items_in_source = new_val
        self._persist()

    # ---- candidate review ----

    def jump_to_candidate(self, idx: int, auto_play: bool = True) -> None:
        """Navigate to candidate `idx`: seek to ts−CAND_LEAD_IN_S. Auto-play
        by default (used by Enter/Backspace navigation); pass auto_play=False
        on episode-switch / first-load so the user gets a still frame and can
        choose when to start watching."""
        if not self.view.has_candidate_workflow():
            return
        idx = max(0, min(len(self.view.candidates) - 1, idx))
        self.view.cand_idx = idx
        c = self.view.candidates[idx]
        self.view.cursor_s = max(0.0, c.ts_s - CAND_LEAD_IN_S)
        self.view.playing = auto_play
        self.view.candidate_paused = False
        self.view.active_field = 'n_successful'
        self._sync_widget_values()

    def advance_candidate(self) -> None:
        """Save current annotation state, then move to the next reviewable
        candidate (skipping NEUTRAL — fumbles back into source bin auto-
        resolve to 0/0 and don't need a click).

        At the last candidate, set cand_idx past-end so `at_episode_end()`
        flips True and the UI highlights the prev/next-episode keys."""
        if not self.view.has_candidate_workflow():
            return
        self._persist()
        nxt = _next_reviewable(self.view, self.view.cand_idx + 1)
        if nxt is None:
            self.view.cand_idx = len(self.view.candidates)
            self.view.playing = False
            self.view.candidate_paused = False
            return
        self.jump_to_candidate(nxt)

    def goto_prev_candidate(self) -> None:
        if not self.view.has_candidate_workflow() or self.view.cand_idx <= 0:
            return
        prev = _prev_reviewable(self.view, self.view.cand_idx - 1)
        if prev is None:
            return
        self.jump_to_candidate(prev)

    def toggle_active_field(self) -> None:
        if not self.view.has_candidate_workflow():
            return
        self.view.active_field = 'n_out_of_bins' if self.view.active_field == 'n_successful' else 'n_successful'
        self._sync_widget_values()

    def set_active_field_value(self, value: int) -> None:
        """Replace the active candidate's active field. Called from digit keys."""
        ann = self.view.active_annotation()
        if ann is None:
            return
        value = max(0, int(value))
        if self.view.active_field == 'n_successful':
            ann.n_successful = value
            if dpg.does_item_exist('inp_n_successful'):
                dpg.set_value('inp_n_successful', value)
        else:
            ann.n_out_of_bins = value
            if dpg.does_item_exist('inp_n_out_of_bins'):
                dpg.set_value('inp_n_out_of_bins', value)
        self._persist()

    def write_widget_value(self, field: str, value: int) -> None:
        """Called from the input_int widgets' on-change callbacks."""
        ann = self.view.active_annotation()
        if ann is None:
            return
        value = max(0, int(value))
        if field == 'n_successful':
            ann.n_successful = value
        elif field == 'n_out_of_bins':
            ann.n_out_of_bins = value
        else:
            return
        self.view.active_field = field
        self._persist()

    def write_comment(self, text: str) -> None:
        ann = self.view.active_annotation()
        if ann is None:
            return
        ann.comment = str(text or '')
        self._persist()

    def _sync_widget_values(self) -> None:
        """Push the active annotation into the widgets without stealing focus.
        We deliberately do NOT call dpg.focus_item — keeping focus on the main
        window means the global digit/Tab/Enter handlers receive keystrokes,
        instead of digits getting appended into the int widget. The yellow
        theme border (bound by _bind_active_theme) is the visible cue for
        which field a digit press will land in. The user can still click into
        a widget to type directly."""
        ann = self.view.active_annotation()
        if ann is None:
            return
        if dpg.does_item_exist('inp_n_successful'):
            dpg.set_value('inp_n_successful', ann.n_successful)
        if dpg.does_item_exist('inp_n_out_of_bins'):
            dpg.set_value('inp_n_out_of_bins', ann.n_out_of_bins)
        if dpg.does_item_exist('inp_comment'):
            dpg.set_value('inp_comment', ann.comment)
        if dpg.does_item_exist('main_window'):
            dpg.focus_item('main_window')

    # ---- playback / scrubbing ----

    def seek_to_candidate(self, direction: int) -> None:
        """Seek the cursor to the next (+1) or previous (-1) candidate
        relative to the current cursor. Used by Enter / Backspace in
        free-scrub-with-overlay mode to jump between stage-1 events
        without the candidate-review annotation flow firing.

        A small epsilon guards against re-jumping to the candidate the
        cursor currently sits on (after a previous jump)."""
        if not self.view.candidates:
            return
        cur = self.view.cursor_s
        ts_sorted = sorted(c.ts_s for c in self.view.candidates)
        eps = 0.05
        if direction > 0:
            target = next((t for t in ts_sorted if t > cur + eps), None)
        else:
            target = next((t for t in reversed(ts_sorted) if t < cur - eps), None)
        if target is None:
            return
        self.view.cursor_s = max(0.0, min(self.view.duration_s, target))
        self.view.playing = False
        self.view.candidate_paused = False

    def seek(self, delta_s: float) -> None:
        self.view.cursor_s = max(0.0, min(self.view.duration_s, self.view.cursor_s + delta_s))
        self.view.playing = False
        self.view.candidate_paused = False

    def toggle_play(self) -> None:
        if self.view.cursor_s >= self.view.duration_s and not self.view.playing:
            self.view.cursor_s = 0.0
        self.view.playing = not self.view.playing
        self.view.candidate_paused = False

    def set_cursor(self, t: float) -> None:
        self.view.cursor_s = max(0.0, min(self.view.duration_s, float(t)))
        self.view.playing = False
        self.view.candidate_paused = False


# ---- key helpers ----


def _seek_amount() -> float:
    shift = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
    alt = dpg.is_key_down(dpg.mvKey_LAlt) or dpg.is_key_down(dpg.mvKey_RAlt)
    if shift and alt:
        return SEEK_BOTH_S
    if alt:
        return SEEK_CTRL_S
    if shift:
        return SEEK_SHIFT_S
    return SEEK_NUDGE_S


def _display_size(shape: tuple[int, int] | None) -> tuple[int, int]:
    """Scale (h, w) to DISPLAY_HEIGHT preserving aspect ratio. Defaults if None."""
    if shape is None:
        return 640, DISPLAY_HEIGHT
    h, w = shape
    if h <= 0:
        return 640, DISPLAY_HEIGHT
    return int(round(w * DISPLAY_HEIGHT / h)), DISPLAY_HEIGHT


# ---- UI construction ----


_DIGIT_KEYS = [
    (dpg.mvKey_0, 0),
    (dpg.mvKey_1, 1),
    (dpg.mvKey_2, 2),
    (dpg.mvKey_3, 3),
    (dpg.mvKey_4, 4),
    (dpg.mvKey_5, 5),
    (dpg.mvKey_6, 6),
    (dpg.mvKey_7, 7),
    (dpg.mvKey_8, 8),
    (dpg.mvKey_9, 9),
]


HELP_LEFT = [
    'CANDIDATE REVIEW (with --candidates)',
    '  Click any field to type directly — fields auto-save.',
    '  0-9         replace active field (when no field focused)',
    '  Tab         toggle active field',
    '  Enter       Save & next candidate',
    '  Backspace   previous candidate',
    '  Comment     click in, type freely; Enter still advances.',
    '',
    'PLAYBACK',
    '  Space       play / pause',
    '  Left/Right  seek (Shift bigger, Alt bigger, both biggest)',
]

HELP_RIGHT = [
    'EPISODE',
    '  , / .       previous / next episode',
    '  R           toggle reviewed (header turns green)',
    '  Q           quit',
    '',
    'FREE SCRUB (no --candidates)',
    '  A           add place at cursor',
    '  X           delete selected place',
    '  D           add drop at cursor (coral tick)',
    '  Shift+D     delete drop nearest cursor',
    '  N / P       next / previous place',
    '  [ / ]       items_in_source_at_end −1 / +1',
]


def _setup_ui(app: App) -> None:
    dpg.create_context()
    app._ensure_textures()

    # Theme: active field gets a vivid yellow border + slightly darkened
    # frame background, so the user always sees which field a digit press
    # would land in. Bound dynamically by `_refresh_candidate_panel`.
    with dpg.theme(tag='active_field_theme'):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Border, (255, 220, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (60, 60, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (90, 90, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (110, 110, 0, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 2.0)

    with dpg.window(label='phail markup', tag='main_window', no_close=True):
        dpg.add_text('', tag='header_text')
        # Top row: cameras side-by-side, no flanking content (max image area).
        ext_w, ext_h = _display_size(app.tex_ext_shape)
        wrist_w, wrist_h = _display_size(app.tex_wrist_shape)
        with dpg.group(horizontal=True):
            dpg.add_image(texture_tag='tex_ext', tag='img_ext', width=ext_w, height=ext_h)
            dpg.add_image(texture_tag='tex_wrist', tag='img_wrist', width=wrist_w, height=wrist_h)
        # Scrubber + timeline span the full width below the cameras.
        dpg.add_slider_float(
            tag='cursor_slider',
            min_value=0.0,
            max_value=app.view.duration_s,
            default_value=0.0,
            width=-1,
            callback=lambda s, a: app.set_cursor(a),
        )
        dpg.add_drawlist(width=-1, height=20, tag='timeline_strip')
        # Compact status line: time, source-counter, save status on one row.
        with dpg.group(horizontal=True):
            dpg.add_text('', tag='time_text')
            dpg.add_spacer(width=24)
            dpg.add_text('', tag='counter_text')
            dpg.add_spacer(width=24)
            dpg.add_text('', tag='status_text')
        dpg.add_text('', tag='places_text', wrap=0)
        dpg.add_separator()

        # Candidate review block.
        dpg.add_text('', tag='cand_header_text')
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text('n_successful', color=DIM_COLOR)
                dpg.add_input_int(
                    tag='inp_n_successful',
                    default_value=0,
                    min_value=0,
                    max_value=99,
                    min_clamped=True,
                    max_clamped=True,
                    width=120,
                    step=1,
                    callback=lambda s, a: app.write_widget_value('n_successful', a),
                )
            dpg.add_spacer(width=24)
            with dpg.group():
                dpg.add_text('n_out_of_bins', color=DIM_COLOR)
                dpg.add_input_int(
                    tag='inp_n_out_of_bins',
                    default_value=0,
                    min_value=0,
                    max_value=99,
                    min_clamped=True,
                    max_clamped=True,
                    width=120,
                    step=1,
                    callback=lambda s, a: app.write_widget_value('n_out_of_bins', a),
                )
            dpg.add_spacer(width=24)
            with dpg.group():
                dpg.add_text('comment (auto-saves)', color=DIM_COLOR)
                dpg.add_input_text(
                    tag='inp_comment', default_value='', width=-1, callback=lambda s, a: app.write_comment(a)
                )
        dpg.add_text('', tag='cand_reasons_text', wrap=0)
        dpg.add_text('', tag='cand_episode_end_text')
        with dpg.group(horizontal=True):
            dpg.add_button(label='Prev cand  (Bksp)', callback=lambda: app.goto_prev_candidate(), tag='btn_prev_cand')
            dpg.add_button(
                label='Save & Next cand  (Enter)', callback=lambda: app.advance_candidate(), tag='btn_next_cand'
            )
            dpg.add_button(label='Play / Pause  (Space)', callback=lambda: app.toggle_play(), tag='btn_play_pause')
            dpg.add_spacer(width=24)
            dpg.add_button(
                label='Prev episode  (,)', callback=lambda: app.switch_episode(app.idx - 1), tag='btn_prev_ep'
            )
            dpg.add_button(
                label='Next episode  (.)', callback=lambda: app.switch_episode(app.idx + 1), tag='btn_next_ep'
            )
            dpg.add_spacer(width=24)
            dpg.add_button(label='Quit  (Q)', callback=lambda: dpg.stop_dearpygui(), tag='btn_quit')

        # Keybindings help — below the actions, two columns.
        dpg.add_spacer(height=12)
        dpg.add_separator()
        dpg.add_spacer(height=4)
        dpg.add_text('KEYBINDINGS', color=ACTIVE_COLOR)
        with dpg.group(horizontal=True):
            with dpg.group():
                for line in HELP_LEFT:
                    color = HINT_COLOR if line and not line.startswith('  ') else DIM_COLOR
                    dpg.add_text(line, color=color)
            dpg.add_spacer(width=40)
            with dpg.group():
                for line in HELP_RIGHT:
                    color = HINT_COLOR if line and not line.startswith('  ') else DIM_COLOR
                    dpg.add_text(line, color=color)

    with dpg.handler_registry():
        dpg.add_key_press_handler(dpg.mvKey_Spacebar, callback=lambda s, a: app.toggle_play())
        dpg.add_key_press_handler(dpg.mvKey_Left, callback=lambda s, a: app.seek(-_seek_amount()))
        dpg.add_key_press_handler(dpg.mvKey_Right, callback=lambda s, a: app.seek(+_seek_amount()))
        dpg.add_key_press_handler(dpg.mvKey_A, callback=lambda s, a: app.add_place())
        dpg.add_key_press_handler(dpg.mvKey_X, callback=lambda s, a: app.delete_selected_place())
        dpg.add_key_press_handler(dpg.mvKey_N, callback=lambda s, a: app.select_place(+1))
        dpg.add_key_press_handler(dpg.mvKey_P, callback=lambda s, a: app.select_place(-1))
        dpg.add_key_press_handler(dpg.mvKey_Open_Brace, callback=lambda s, a: app.adjust_source(-1))
        dpg.add_key_press_handler(dpg.mvKey_Close_Brace, callback=lambda s, a: app.adjust_source(+1))
        dpg.add_key_press_handler(dpg.mvKey_Comma, callback=lambda s, a: app.switch_episode(app.idx - 1))
        dpg.add_key_press_handler(dpg.mvKey_Period, callback=lambda s, a: app.switch_episode(app.idx + 1))
        dpg.add_key_press_handler(dpg.mvKey_Q, callback=lambda s, a: dpg.stop_dearpygui())
        dpg.add_key_press_handler(dpg.mvKey_R, callback=lambda s, a: app.toggle_reviewed())
        # Free-scrub drop events: D adds a drop at the cursor (coral tick on
        # the timeline). Shift+D deletes the drop nearest the cursor. Inert
        # in candidate-review mode (n_out_of_bins captures the same info
        # per candidate).
        def _drop_key(_s, _a):
            if dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift):
                app.delete_nearest_drop()
            else:
                app.add_drop()

        dpg.add_key_press_handler(dpg.mvKey_D, callback=_drop_key)
        # Candidate review. Gate keys whose meaning would conflict with
        # typing into the comment field — Enter/Tab/Backspace + digits all
        # have natural in-text meaning.
        dpg.add_key_press_handler(dpg.mvKey_Return, callback=_make_advance_callback(app))
        dpg.add_key_press_handler(dpg.mvKey_Tab, callback=_make_toggle_callback(app))
        dpg.add_key_press_handler(dpg.mvKey_Back, callback=_make_prev_cand_callback(app))
        for key, digit in _DIGIT_KEYS:
            dpg.add_key_press_handler(key, callback=_make_digit_callback(app, digit))

    dpg.create_viewport(title='phail markup', width=1480, height=960)
    dpg.set_primary_window('main_window', True)
    dpg.setup_dearpygui()
    dpg.show_viewport()


def _comment_is_focused() -> bool:
    return dpg.does_item_exist('inp_comment') and dpg.is_item_focused('inp_comment')


def _int_field_is_focused() -> bool:
    for tag in ('inp_n_successful', 'inp_n_out_of_bins'):
        if dpg.does_item_exist(tag) and dpg.is_item_focused(tag):
            return True
    return False


def _make_digit_callback(app: App, digit: int):
    """Replace App.active_field with `digit`. Skip when the comment field
    or an int field has dearpygui focus — in those cases the user is
    typing directly and dearpygui handles the key."""

    def _cb(_sender, _app_data):
        if not app.view.has_candidate_workflow():
            return
        if _comment_is_focused() or _int_field_is_focused():
            return
        app.set_active_field_value(digit)

    return _cb


def _make_advance_callback(app: App):
    def _cb(_sender, _app_data):
        # Enter inside the comment field still advances — the user signals
        # they're done with this candidate. The comment is already auto-saved.
        if _int_field_is_focused():
            # Drop focus first so dearpygui finalizes the value.
            dpg.focus_item('main_window')
        if app.view.has_candidate_workflow():
            app.advance_candidate()
        elif app.view.has_candidate_overlay():
            app.seek_to_candidate(+1)

    return _cb


def _make_toggle_callback(app: App):
    def _cb(_sender, _app_data):
        if _comment_is_focused():
            return
        app.toggle_active_field()

    return _cb


def _make_prev_cand_callback(app: App):
    def _cb(_sender, _app_data):
        if _comment_is_focused() or _int_field_is_focused():
            return
        if app.view.has_candidate_workflow():
            app.goto_prev_candidate()
        elif app.view.has_candidate_overlay():
            app.seek_to_candidate(-1)

    return _cb


def _format_header(app: App) -> str:
    m = app.view.meta
    badge = '  [REVIEWED]' if app.view.reviewed else ''
    return (
        f'ep {app.idx + 1}/{len(app.episodes)}  {app.view.ref.rel_id}  '
        f'obj={m.get("eval.object", "?")}  N={m.get("eval.total_items", "?")}  '
        f'outcome={m.get("eval.outcome", "?")}  '
        f'model={m.get("model", "?")}  '
        f'ref-S={m.get("eval.successful_items", "?")}'
        f'{badge}'
    )


def _format_places(app: App) -> str:
    if not app.view.place_timestamps_s:
        return 'places (0): (none)'
    parts = []
    for i, t in enumerate(app.view.place_timestamps_s):
        if i == app.view.selected_idx:
            parts.append(f'[{t:.2f}s]')
        else:
            parts.append(f'{t:.2f}s')
    return f'places ({len(app.view.place_timestamps_s)}): ' + '  '.join(parts)


def _format_counter(app: App) -> str:
    n = app.view.total_items
    target = len(app.view.place_timestamps_s)
    source = app.view.items_in_source
    drops = n - target - source
    over = '  (OVER N!)' if (target + source) > n else ''
    return (
        f'items_in_source_at_end={source} (use [/])  '
        f'items_in_target_at_end={target} (auto)  '
        f'drops={drops} (derived)  N={n}{over}'
    )


def _format_status(app: App) -> str:
    if app.last_save_t is None:
        return 'status: clean (no edits yet)'
    age = time.monotonic() - app.last_save_t
    return f'status: saved {age:.1f}s ago'


def _redraw_timeline(app: App) -> None:
    if not dpg.does_item_exist('timeline_strip'):
        return
    try:
        vp_w = dpg.get_viewport_client_width()
    except Exception:
        vp_w = 1300
    width = max(200, vp_w - 40)  # span almost full window width (small padding)
    height = 20
    if dpg.get_item_width('timeline_strip') != width:
        dpg.configure_item('timeline_strip', width=width, height=height)

    dpg.delete_item('timeline_strip', children_only=True)
    dur = max(app.view.duration_s, 1e-3)
    y_mid = height // 2
    dpg.draw_line((0, y_mid), (width, y_mid), color=(80, 80, 80, 255), thickness=1, parent='timeline_strip')
    cx = int(app.view.cursor_s / dur * width)
    dpg.draw_line((cx, 0), (cx, height), color=(255, 220, 0, 255), thickness=2, parent='timeline_strip')
    # Stage-1 candidate ticks (light blue) — drawn whenever a candidates file is
    # loaded. In candidate-review workflow the active candidate is highlighted
    # pink; in free-scrub-with-overlay mode the active-candidate concept doesn't
    # apply and they all render uniformly.
    if app.view.has_candidate_workflow():
        for i, c in enumerate(app.view.candidates):
            x = int(c.ts_s / dur * width)
            selected = i == app.view.cand_idx
            color = (255, 80, 200, 255) if selected else (80, 200, 255, 255)
            thickness = 4 if selected else 2
            dpg.draw_line((x, 0), (x, height), color=color, thickness=thickness, parent='timeline_strip')
    elif app.view.has_candidate_overlay():
        for c in app.view.candidates:
            x = int(c.ts_s / dur * width)
            dpg.draw_line((x, 0), (x, height), color=(80, 200, 255, 255), thickness=2, parent='timeline_strip')
    # Manual place ticks: green so they're visually distinct from the
    # light-blue candidate ticks they may sit directly on top of in
    # free-scrub-overlay mode (saw a "place not visible" report when a
    # manual tick landed on the same x-pixel as a candidate). Drawn at
    # double thickness so even when stacked the green dominates.
    if not app.view.has_candidate_workflow():
        for i, t in enumerate(app.view.place_timestamps_s):
            x = int(t / dur * width)
            selected = i == app.view.selected_idx
            color = (255, 80, 200, 255) if selected else (60, 220, 110, 255)
            thickness = 5 if selected else 4
            dpg.draw_line((x, 0), (x, height), color=color, thickness=thickness, parent='timeline_strip')
    # Free-scrub drops (red ticks). Rendered in both modes — they read from
    # disk and may be present on legacy files even in candidate-review mode.
    for t in app.view.drop_timestamps_s:
        x = int(t / dur * width)
        dpg.draw_line((x, 0), (x, height), color=(235, 80, 80, 255), thickness=3, parent='timeline_strip')


def _refresh_text(app: App) -> None:
    dpg.set_value('header_text', _format_header(app))
    dpg.configure_item(
        'header_text',
        color=(80, 220, 120) if app.view.reviewed else INACTIVE_COLOR,
    )
    dpg.set_value(
        'time_text',
        f't = {app.view.cursor_s:6.2f}s / {app.view.duration_s:6.2f}s   {"PLAY" if app.view.playing else "pause"}',
    )
    dpg.set_value('places_text', _format_places(app))
    dpg.set_value('counter_text', _format_counter(app))
    dpg.set_value('status_text', _format_status(app))
    if dpg.does_item_exist('cursor_slider'):
        dpg.configure_item('cursor_slider', max_value=app.view.duration_s)
        dpg.set_value('cursor_slider', app.view.cursor_s)
    _refresh_candidate_panel(app)
    _redraw_timeline(app)


def _bind_active_theme(app: App) -> None:
    """Bind the active-field theme to whichever int field is the App's
    active_field, and clear it from the other. The theme draws a yellow
    border so the user always sees which field a digit press will land in
    (independent of dearpygui's transient focus indicator)."""
    if not app.view.has_candidate_workflow():
        return
    is_succ = app.view.active_field == 'n_successful'
    if dpg.does_item_exist('inp_n_successful'):
        dpg.bind_item_theme('inp_n_successful', 'active_field_theme' if is_succ else 0)
    if dpg.does_item_exist('inp_n_out_of_bins'):
        dpg.bind_item_theme('inp_n_out_of_bins', 'active_field_theme' if not is_succ else 0)


def _set_panel_disabled(disabled: bool) -> None:
    for tag in ('inp_n_successful', 'inp_n_out_of_bins', 'inp_comment'):
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, enabled=not disabled)


def _refresh_candidate_panel(app: App) -> None:
    if not app.view.has_candidate_workflow():
        dpg.set_value('cand_header_text', '(no --candidates given — free-scrub mode)')
        dpg.configure_item('cand_header_text', color=DIM_COLOR)
        _set_panel_disabled(True)
        dpg.set_value('cand_reasons_text', '')
        dpg.set_value('cand_episode_end_text', '')
        return

    n_total = len(app.view.candidates)
    cand = app.view.active_candidate()
    ann = app.view.active_annotation()

    if cand is None or ann is None:
        # End of episode — disable inputs, highlight the prev/next-ep buttons.
        dpg.set_value('cand_header_text', f'episode complete — reviewed all {n_total} candidates')
        dpg.configure_item('cand_header_text', color=ACTIVE_COLOR)
        _set_panel_disabled(True)
        dpg.set_value('cand_reasons_text', '')
        dpg.set_value('cand_episode_end_text', 'press , for previous episode    or    press . for next episode')
        dpg.configure_item('cand_episode_end_text', color=HINT_HIGHLIGHT_COLOR)
        for tag in ('btn_prev_ep', 'btn_next_ep'):
            if dpg.does_item_exist(tag):
                dpg.bind_item_theme(tag, 'active_field_theme')
        return

    # Active candidate.
    _set_panel_disabled(False)
    dpg.set_value(
        'cand_header_text',
        f'candidate {app.view.cand_idx + 1}/{n_total}  '
        f'@ {cand.ts_s:.2f}s  '
        f'verdict={cand.verdict.upper()}  kind={cand.kind}',
    )
    dpg.configure_item('cand_header_text', color=ACTIVE_COLOR)
    reasons = ', '.join(cand.reasons) if cand.reasons else '(none)'
    dpg.set_value(
        'cand_reasons_text', f'reasons: {reasons}    [Tab toggles field, 0-9 replaces active, Enter saves & advances]'
    )
    dpg.configure_item('cand_reasons_text', color=HINT_COLOR)
    dpg.set_value('cand_episode_end_text', '')
    for tag in ('btn_prev_ep', 'btn_next_ep'):
        if dpg.does_item_exist(tag):
            dpg.bind_item_theme(tag, 0)
    _bind_active_theme(app)


# ---- main loop ----


def _step_main_loop(app: App, dt: float) -> None:
    """One tick of state advance: playback + candidate auto-pause."""
    if app.view.playing:
        app.view.cursor_s += dt
        if app.view.cursor_s >= app.view.duration_s:
            app.view.cursor_s = app.view.duration_s
            app.view.playing = False

    cand = app.view.active_candidate()
    if app.view.playing and cand is not None and not app.view.candidate_paused:
        pause_at = cand.ts_s + CAND_LEAD_OUT_S
        if app.view.cursor_s >= pause_at:
            app.view.cursor_s = min(pause_at, app.view.duration_s)
            app.view.playing = False
            app.view.candidate_paused = True


def run_ui(
    episodes: list[EpisodeRef],
    out_dir: Path,
    candidates_by_ep: dict[tuple[str, str], list[Candidate]] | None,
    free_scrub: bool = False,
) -> None:
    app = App(episodes, out_dir, candidates_by_ep, free_scrub=free_scrub)
    _setup_ui(app)
    if app.view.has_candidate_workflow():
        first = _first_reviewable(app.view)
        if first is not None:
            app.jump_to_candidate(first, auto_play=False)
        else:
            # All candidates auto-resolved (NEUTRAL only) — show end-of-episode hint.
            app.view.cand_idx = len(app.view.candidates)

    last_t = time.monotonic()
    while dpg.is_dearpygui_running():
        now = time.monotonic()
        dt = now - last_t
        last_t = now
        _step_main_loop(app, dt)

        app.refresh_frames()
        _refresh_text(app)
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


# ---- episode resolution ----


def _join_remote(root: str, sub: str) -> str:
    return f'{root.rstrip("/")}/{sub.lstrip("/")}'


def _derive_source(root: str) -> str:
    """Pick the dataset source bucket from the trailing path segment of
    cache_root or episode_root. The dataset layout is `.../<source>/<rel>`
    where source ∈ {inference, training, human}; falling back to 'inference'
    matches the historical default."""
    seg = root.rstrip('/').rsplit('/', 1)[-1]
    return seg if seg in ('inference', 'training', 'human') else 'inference'


def resolve_episodes(debug_set: str | None, episode_root: str | None, cache_root: str | None) -> list[EpisodeRef]:
    if debug_set is None and episode_root is None:
        raise SystemExit('provide --debug_set or --episode_root')
    if debug_set is not None and episode_root is not None:
        log.warning('both --debug_set and --episode_root given; --debug_set wins')

    if debug_set is not None:
        # debug_set may be local (no profile needed) or s3:// (needs the
        # public profile for credential-free S3 access).
        debug_local = (
            pos3.download(debug_set, profile=PUBLIC_PROFILE)
            if str(debug_set).startswith('s3://')
            else pos3.download(debug_set)
        )
        with open(debug_local) as f:
            entries = json.load(f)
        root = cache_root or DEFAULT_CACHE_ROOT
        source = _derive_source(root)
        log.info('resolving %d episodes from %s (source=%s)', len(entries), root, source)
        out = []
        for e in entries:
            rel_id = e['ep_dir'].strip('/')
            abs_path = Path(pos3.download(_join_remote(root, rel_id), profile=PUBLIC_PROFILE))
            out.append(EpisodeRef(abs_path=abs_path, rel_id=rel_id, source=source))
        return out

    root_local = Path(
        pos3.download(episode_root, profile=PUBLIC_PROFILE)
        if str(episode_root).startswith('s3://')
        else pos3.download(episode_root)
    )
    source = _derive_source(episode_root)
    log.info('resolving episodes from %s (source=%s)', root_local, source)
    out = []
    for static_path in sorted(root_local.rglob('static.json')):
        ep_dir = static_path.parent
        rel_id = ep_dir.relative_to(root_local).as_posix()
        out.append(EpisodeRef(abs_path=ep_dir, rel_id=rel_id, source=source))
    return out


# ---- entry point ----


@cfn.config(debug_set=None, episode_root=None, cache_root=None, candidates=None, out_dir=DEFAULT_OUT_DIR, free_scrub=False)
def main(
    debug_set: str | None,
    episode_root: str | None,
    cache_root: str | None,
    candidates: str | None,
    out_dir: str,
    free_scrub: bool,
) -> None:
    episodes = resolve_episodes(debug_set, episode_root, cache_root)
    if not episodes:
        raise SystemExit('no episodes resolved')
    candidates_by_ep: dict[tuple[str, str], list[Candidate]] | None = None
    if candidates:
        cand_path = Path(candidates).expanduser().resolve()
        if not cand_path.exists():
            raise SystemExit(f'--candidates {cand_path} not found')
        candidates_by_ep = _load_candidates_by_episode(cand_path)
        per_source: dict[str, int] = {}
        for src, _ in candidates_by_ep:
            per_source[src] = per_source.get(src, 0) + 1
        log.info(
            'loaded candidates for %d (source, ep) keys from %s — %s%s',
            len(candidates_by_ep),
            cand_path,
            ', '.join(f'{s}={n}' for s, n in sorted(per_source.items())),
            '  (overlay only — --free_scrub set)' if free_scrub else '',
        )
    log.info('Loaded %d episodes; markup -> %s', len(episodes), out_dir)
    run_ui(episodes, Path(out_dir).expanduser().resolve(), candidates_by_ep, free_scrub=free_scrub)


@pos3.with_mirror()
def _internal_main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfn.cli(main)


if __name__ == '__main__':
    _internal_main()
