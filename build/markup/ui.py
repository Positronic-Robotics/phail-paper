"""Manual markup UI for D9 episode logs.

Annotate each successful "place" event in an inference episode by scrubbing
through the wrist + exterior cameras with keyboard nudges. Edits auto-save
to a sidecar JSON file outside the source dataset, keyed by the episode's
relative id (e.g. "000000000000/000000000293.json").

Episode list is provided via either:
  --debug_set  build/autopilot/debug_set.json  (curated)
  --episode_root  <dir>                         (walk a tree)

Both s3:// and local paths work via pos3.

Usage:
    uv run python -m build.markup.ui --debug_set build/autopilot/debug_set.json
    uv run python -m build.markup.ui --episode_root ~/.cache/positronic/.../inference
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import configuronic as cfn
import dearpygui.dearpygui as dpg
import numpy as np
import pos3
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
DISPLAY_HEIGHT = 360
SOURCE_DEC_KEY = '['
SOURCE_INC_KEY = ']'
PREV_EP_KEY = ','
NEXT_EP_KEY = '.'


@dataclass
class EpisodeRef:
    abs_path: Path
    rel_id: str  # e.g. "000000000000/000000000293"


class EpisodeView:
    """Per-episode mutable state."""

    def __init__(self, ref: EpisodeRef, out_dir: Path):
        self.ref = ref
        self.markup_path = markup_io.markup_path(out_dir, ref.rel_id)
        self.ep = DiskEpisode(ref.abs_path)
        self.start_ts = int(self.ep.start_ts)
        self.last_ts = int(self.ep.last_ts)
        self.duration_s = self.ep.duration_ns / 1e9
        self.video_wrist = self.ep['image.wrist']
        self.video_ext = self.ep['image.exterior']
        self.place_timestamps_s, self.items_in_source = markup_io.load_markup(self.markup_path)
        self.selected_idx: int | None = 0 if self.place_timestamps_s else None
        self.cursor_s = 0.0
        self.playing = False

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


class App:
    def __init__(self, episodes: list[EpisodeRef], out_dir: Path):
        if not episodes:
            raise ValueError('no episodes provided')
        self.episodes = episodes
        self.out_dir = out_dir
        self.idx = 0
        self.view: EpisodeView = EpisodeView(episodes[0], out_dir)
        self.tex_wrist_shape: tuple[int, int] | None = None
        self.tex_ext_shape: tuple[int, int] | None = None
        self.tex_wrist: np.ndarray | None = None
        self.tex_ext: np.ndarray | None = None
        self.last_save_t: float | None = None
        self.last_render_t = time.monotonic()

    # ---- episode lifecycle ----

    def switch_episode(self, new_idx: int) -> None:
        new_idx = max(0, min(len(self.episodes) - 1, new_idx))
        if new_idx == self.idx:
            return
        self.idx = new_idx
        self.view = EpisodeView(self.episodes[new_idx], self.out_dir)
        self._ensure_textures()
        self.last_save_t = None

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
        markup_io.save_markup(
            self.view.markup_path,
            self.view.place_timestamps_s,
            self.view.items_in_source,
        )
        self.last_save_t = time.monotonic()

    def add_place(self) -> None:
        self.view.place_timestamps_s.append(round(self.view.cursor_s, 3))
        self.view.place_timestamps_s.sort()
        self.view.selected_idx = self.view.place_timestamps_s.index(
            round(self.view.cursor_s, 3)
        )
        self._persist()

    def delete_selected_place(self) -> None:
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
            self.view.selected_idx = (self.view.selected_idx + delta) % len(
                self.view.place_timestamps_s
            )
        self.view.cursor_s = self.view.place_timestamps_s[self.view.selected_idx]
        self.view.playing = False

    def adjust_source(self, delta: int) -> None:
        new_val = self.view.items_in_source + delta
        new_val = max(0, min(self.view.total_items, new_val))
        if new_val == self.view.items_in_source:
            return
        self.view.items_in_source = new_val
        self._persist()

    # ---- playback / scrubbing ----

    def seek(self, delta_s: float) -> None:
        self.view.cursor_s = max(
            0.0, min(self.view.duration_s, self.view.cursor_s + delta_s)
        )
        self.view.playing = False

    def toggle_play(self) -> None:
        if self.view.cursor_s >= self.view.duration_s and not self.view.playing:
            self.view.cursor_s = 0.0
        self.view.playing = not self.view.playing

    def set_cursor(self, t: float) -> None:
        self.view.cursor_s = max(0.0, min(self.view.duration_s, float(t)))
        self.view.playing = False


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


def _setup_ui(app: App) -> None:
    dpg.create_context()
    app._ensure_textures()

    with dpg.window(label='phail markup', tag='main_window', no_close=True):
        dpg.add_text('', tag='header_text')
        with dpg.group(horizontal=True):
            ext_w, ext_h = _display_size(app.tex_ext_shape)
            wrist_w, wrist_h = _display_size(app.tex_wrist_shape)
            dpg.add_image(texture_tag='tex_ext', tag='img_ext', width=ext_w, height=ext_h)
            dpg.add_image(texture_tag='tex_wrist', tag='img_wrist', width=wrist_w, height=wrist_h)
        dpg.add_slider_float(
            tag='cursor_slider',
            min_value=0.0,
            max_value=app.view.duration_s,
            default_value=0.0,
            width=-1,
            callback=lambda s, a: app.set_cursor(a),
        )
        dpg.add_drawlist(width=1300, height=24, tag='timeline_strip')
        dpg.add_text('', tag='time_text')
        dpg.add_text('', tag='places_text', wrap=0)
        dpg.add_text('', tag='counter_text')
        dpg.add_text('', tag='status_text')

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

    dpg.create_viewport(title='phail markup', width=1400, height=900)
    dpg.set_primary_window('main_window', True)
    dpg.setup_dearpygui()
    dpg.show_viewport()


def _format_header(app: App) -> str:
    m = app.view.meta
    return (
        f'ep {app.idx + 1}/{len(app.episodes)}  {app.view.ref.rel_id}  '
        f'obj={m.get("eval.object", "?")}  N={m.get("eval.total_items", "?")}  '
        f'outcome={m.get("eval.outcome", "?")}  '
        f'model={m.get("model", "?")}  '
        f'ref-S={m.get("eval.successful_items", "?")}'
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
    width = max(200, vp_w - 60)
    height = 24
    if dpg.get_item_width('timeline_strip') != width:
        dpg.configure_item('timeline_strip', width=width, height=height)

    dpg.delete_item('timeline_strip', children_only=True)
    dur = max(app.view.duration_s, 1e-3)
    y_mid = height // 2
    dpg.draw_line((0, y_mid), (width, y_mid), color=(80, 80, 80, 255),
                  thickness=1, parent='timeline_strip')
    cx = int(app.view.cursor_s / dur * width)
    dpg.draw_line((cx, 0), (cx, height), color=(255, 220, 0, 255),
                  thickness=2, parent='timeline_strip')
    for i, t in enumerate(app.view.place_timestamps_s):
        x = int(t / dur * width)
        selected = i == app.view.selected_idx
        color = (255, 80, 200, 255) if selected else (80, 200, 255, 255)
        thickness = 4 if selected else 3
        dpg.draw_line((x, 0), (x, height), color=color,
                      thickness=thickness, parent='timeline_strip')


def _refresh_text(app: App) -> None:
    dpg.set_value('header_text', _format_header(app))
    dpg.set_value(
        'time_text',
        f't = {app.view.cursor_s:6.2f}s / {app.view.duration_s:6.2f}s   '
        f'{"PLAY" if app.view.playing else "pause"}',
    )
    dpg.set_value('places_text', _format_places(app))
    dpg.set_value('counter_text', _format_counter(app))
    dpg.set_value('status_text', _format_status(app))
    if dpg.does_item_exist('cursor_slider'):
        dpg.configure_item('cursor_slider', max_value=app.view.duration_s)
        dpg.set_value('cursor_slider', app.view.cursor_s)
    _redraw_timeline(app)


# ---- main loop ----


def run_ui(episodes: list[EpisodeRef], out_dir: Path) -> None:
    app = App(episodes, out_dir)
    _setup_ui(app)

    last_t = time.monotonic()
    while dpg.is_dearpygui_running():
        now = time.monotonic()
        dt = now - last_t
        last_t = now
        if app.view.playing:
            app.view.cursor_s += dt
            if app.view.cursor_s >= app.view.duration_s:
                app.view.cursor_s = app.view.duration_s
                app.view.playing = False

        app.refresh_frames()
        _refresh_text(app)
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


# ---- episode resolution ----


def _join_remote(root: str, sub: str) -> str:
    return f'{root.rstrip("/")}/{sub.lstrip("/")}'


def resolve_episodes(
    debug_set: str | None,
    episode_root: str | None,
    cache_root: str | None,
) -> list[EpisodeRef]:
    if debug_set is None and episode_root is None:
        raise SystemExit('provide --debug_set or --episode_root')
    if debug_set is not None and episode_root is not None:
        log.warning('both --debug_set and --episode_root given; --debug_set wins')

    if debug_set is not None:
        debug_local = pos3.download(debug_set)
        with open(debug_local) as f:
            entries = json.load(f)
        root = cache_root or DEFAULT_CACHE_ROOT
        log.info('resolving %d episodes from %s', len(entries), root)
        out = []
        for e in entries:
            rel_id = e['ep_dir'].strip('/')
            abs_path = Path(pos3.download(_join_remote(root, rel_id)))
            out.append(EpisodeRef(abs_path=abs_path, rel_id=rel_id))
        return out

    root_local = Path(pos3.download(episode_root))
    out = []
    for static_path in sorted(root_local.rglob('static.json')):
        ep_dir = static_path.parent
        rel_id = ep_dir.relative_to(root_local).as_posix()
        out.append(EpisodeRef(abs_path=ep_dir, rel_id=rel_id))
    return out


# ---- entry point ----


@cfn.config(debug_set=None, episode_root=None, cache_root=None, out_dir=DEFAULT_OUT_DIR)
def main(
    debug_set: str | None,
    episode_root: str | None,
    cache_root: str | None,
    out_dir: str,
) -> None:
    episodes = resolve_episodes(debug_set, episode_root, cache_root)
    if not episodes:
        raise SystemExit('no episodes resolved')
    log.info('Loaded %d episodes; markup -> %s', len(episodes), out_dir)
    run_ui(episodes, Path(out_dir).expanduser().resolve())


@pos3.with_mirror()
def _internal_main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfn.cli(main)


if __name__ == '__main__':
    _internal_main()
