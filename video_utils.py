"""
Functional video-building library for wake-model variables.

Collects the animation boilerplate repeated across shapiro_unsteady.py,
multi_forcing.py, super_position.py, varying_turbine.py and
multi_turbine_diff.py into three reusable views built on a common
WakeSeries data container:

    field_video(series, filename)      -- just the 2D flow field + centerline
    profiles_video(series, filename)   -- just the u1 / u2 / yc profiles
    full_video(series, filename)       -- field and profiles together

`series` may be a single WakeSeries or a list of them (e.g. one per turbine).
With a list, field panels sum the individual fields into the total (deficits
superpose additively, as elsewhere in this codebase) and draw one centreline
per series; profile panels overlay one line per series. Each function also
takes an optional label_fn(i) -> str for the per-frame annotation; if not
given, one is built from each series' Turbine yaw (via yaw_label_fn).

Pass diff=True (with a series built via diff_series) to render a
difference between two runs, e.g. multi-forcing vs. superposition.
"""

import os
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator, PercentFormatter

from wake_dynamics import u_point
from unsteady_flow_solver import Turbine

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def make_field_frames(x_grid, yc_xt, u1_xt, y_grid, ts, wp):
    build_frame = jax.vmap(u_point, in_axes=(0, 0, 0, None, None, None), out_axes=1)
    all_frames = jax.vmap(build_frame, in_axes=(None, 0, 0, None, 0, None), out_axes=0)(x_grid, yc_xt, u1_xt, y_grid, ts, wp)
    assert all_frames.shape == (u1_xt.shape[0], y_grid.size, x_grid.size), all_frames.shape
    return np.asarray(all_frames)


def save_video(animation, filename, dpi=120, fps=20):

    filepath = OUT_DIR / filename
    bar = tqdm(desc=filename, unit="frame")

    def progress(i, n):
        bar.total = n
        bar.update(1)

    animation.save(str(filepath), writer='ffmpeg', dpi=dpi, fps=fps, progress_callback=progress)
    bar.close()

    assert filepath.exists(), f"save produced no file: {filepath}"
    os.startfile(filepath)


@dataclass
class WakeSeries:
    """A time series of the three flow variables over a shared x grid, ready to animate."""
    x_grid:   ArrayLike              # (nx,)
    ts:       ArrayLike              # (nt,)
    u1_xt:    ArrayLike              # (nt, nx)
    u2_xt:    ArrayLike              # (nt, nx)
    yc_xt:    ArrayLike              # (nt, nx)
    tb:       Turbine                # the turbine this series belongs to (wp incl. gamma_fn, plus x0)
    frames:   Optional[np.ndarray] = None   # (nt, ny, nx) -- required by field_video/full_video
    y_grid:   Optional[np.ndarray] = None   # paired with frames

    def __post_init__(self):
        self.x_grid = np.asarray(self.x_grid)
        self.ts = np.asarray(self.ts)
        self.u1_xt = np.asarray(self.u1_xt)
        self.u2_xt = np.asarray(self.u2_xt)
        self.yc_xt = np.asarray(self.yc_xt)


SeriesLike = Union[WakeSeries, list[WakeSeries], tuple[WakeSeries, ...]]


@dataclass
class SteadyRef:
    """A time-invariant reference solution overlaid as a dashed red line."""
    u1_x: np.ndarray
    u2_x: np.ndarray
    yc_x: np.ndarray
    label: str = "Steady solution"


def with_field(series: SeriesLike, y_grid) -> SeriesLike:
    """Return a copy of series (or of each series in a list) with flow-field frames computed over y_grid."""
    if isinstance(series, (list, tuple)):
        return [with_field(s, y_grid) for s in series]
    frames = make_field_frames(series.x_grid, series.yc_xt, series.u1_xt, y_grid, series.ts, series.tb.wp)
    return dataclasses.replace(series, frames=frames, y_grid=np.asarray(y_grid))


def diff_series(a: WakeSeries, b: WakeSeries, *, u1_scale=1.0, u2_scale=1.0, yc_scale=1.0) -> WakeSeries:
    """Elementwise a - b, optionally normalised per-variable (e.g. by envelope magnitudes)."""
    assert np.allclose(a.x_grid, b.x_grid), "series must share the same x grid"
    assert np.allclose(a.ts, b.ts), "series must share the same time grid"

    print(f"max |du1|/du1_env = {np.max(np.abs(np.asarray(a.u1_xt) - np.asarray(b.u1_xt)) / u1_scale):.2e}")
    print(f"max |du2|/du2_env = {np.max(np.abs(np.asarray(a.u2_xt) - np.asarray(b.u2_xt)) / u2_scale):.2e}")
    print(f"max |dyc|/yc_max = {np.max(np.abs(np.asarray(a.yc_xt) - np.asarray(b.yc_xt)) / yc_scale):.2e}")

    return WakeSeries(
        x_grid=a.x_grid, ts=a.ts, tb=a.tb,
        u1_xt=(np.asarray(a.u1_xt) - np.asarray(b.u1_xt)) / u1_scale,
        u2_xt=(np.asarray(a.u2_xt) - np.asarray(b.u2_xt)) / u2_scale,
        yc_xt=(np.asarray(a.yc_xt) - np.asarray(b.yc_xt)) / yc_scale,
    )


def yaw_label_fn(params, ts) -> Callable[[int], str]:
    """Per-frame label showing yaw angle(s) alongside time, for use as a video-function label_fn.

    `params` is a list of WakeParams-like objects (each needs .gamma_at(t)) -- one entry
    for a single turbine, or one per turbine to label each separately, e.g.:

        yaw_label_fn([wp], sp.ts)                          # single, static or time-varying yaw
        yaw_label_fn([tb.wp for tb in turbines], sp.ts)     # one subscripted label per turbine
    """
    def label(i):
        t = ts[i]
        yaws = [float(jnp.degrees(p.gamma_at(t))) for p in params]
        if len(yaws) == 1:
            return rf"$\gamma$ = {yaws[0]:.1f}°, t = {t:.0f} s"
        yaw_str = ", ".join(rf"$\gamma_{{{j+1}}}$={y:.1f}°" for j, y in enumerate(yaws))
        return rf"{yaw_str}, t = {t:.0f} s"
    return label


def _as_list(series: SeriesLike) -> list:
    return list(series) if isinstance(series, (list, tuple)) else [series]


def _check_shared_grid(series_list: list):
    x0, ts0 = series_list[0].x_grid, series_list[0].ts
    for s in series_list[1:]:
        assert np.allclose(s.x_grid, x0), "all series must share the same x grid"
        assert np.allclose(s.ts, ts0), "all series must share the same time grid"


def _resolve_label_fn(series_list: list, label_fn: Optional[Callable[[int], str]]) -> Callable[[int], str]:
    """Use the given label_fn if provided, else build one from each series' Turbine yaw."""
    if label_fn is not None:
        return label_fn
    return yaw_label_fn([s.tb.wp for s in series_list], series_list[0].ts)


def _add_gridlines(axes):
    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(5.0))
        ax.xaxis.set_minor_locator(MultipleLocator(1.0))
        ax.grid(True, which='major', axis='x', lw=0.6, alpha=0.5)
        ax.grid(True, which='minor', axis='x', lw=0.3, alpha=0.25)


def _pad_ylim(ax, *arrays, diff=False, pad=0.1):
    lo = min(float(np.min(a)) for a in arrays)
    hi = max(float(np.max(a)) for a in arrays)
    if diff:
        m = (1 + pad) * max(abs(lo), abs(hi))
        ax.set_ylim(-m, m)
    else:
        span = hi - lo or 1.0
        ax.set_ylim(lo - pad * span, hi + pad * span)


def _draw_field(ax, fig, series_list: list, *, diff: bool, percent: bool,
                 steady: Optional[SteadyRef], colorbar_loc='right', colorbar_shrink=None):
    multi = len(series_list) > 1
    wp = series_list[0].tb.wp
    x_norm = series_list[0].x_grid / wp.D
    y_norm = series_list[0].y_grid / wp.D
    frames = np.sum(np.stack([s.frames for s in series_list], axis=0), axis=0)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    cbar_kwargs = dict(location=colorbar_loc)
    if colorbar_shrink is not None:
        cbar_kwargs['shrink'] = colorbar_shrink

    if diff:
        m = np.abs(frames).max()
        mesh = ax.pcolormesh(x_norm, y_norm, frames[0], cmap='RdBu_r', shading='auto', vmin=-m, vmax=m)
        if percent:
            cbar_kwargs['format'] = PercentFormatter(xmax=1.0)
            cbar_kwargs['label'] = r'$\Delta u_1/\delta u_{1,\mathrm{env}}$'
        else:
            cbar_kwargs['label'] = r'$\Delta u_1$'
    else:
        mesh = ax.pcolormesh(x_norm, y_norm, frames[0], cmap='plasma', shading='auto', vmin=0, vmax=frames.max())
        cbar_kwargs['label'] = r'$\delta u_1$ [m/s]'
        cl_label = 'Unsteady $y_c$' if steady is not None else '$y_c$'

    fig.colorbar(mesh, ax=ax, **cbar_kwargs)

    # centrelines don't add information on a diff plot (there's no single "the" wake
    # to trace) -- only draw them for the non-diff field view, colour-matched to each
    # turbine's profile lines below so it's clear which centreline is which turbine.
    cl_lines = []
    if not diff:
        cl_lines = [
            ax.plot(x_norm, s.yc_xt[0] / wp.D, '--', color=colors[i % len(colors)], lw=1.2,
                    label=f'Turbine {i+1}' if multi else cl_label)[0]
            for i, s in enumerate(series_list)
        ]

    if steady is not None:
        ax.plot(x_norm, steady.yc_x / wp.D, 'r--', lw=1.0, label=steady.label)

    ax.set_ylabel(r'$y/D$')
    if cl_lines or steady is not None:
        ax.legend(loc='upper right')
    # bbox guarantees the label reads over both the dark plasma colormap and the
    # near-white centre of the diverging RdBu_r diff colormap.
    timestamp = ax.text(0.02, 0.92, '', transform=ax.transAxes, color='w',
                         bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.45, edgecolor='none'))
    return mesh, cl_lines, frames, timestamp


def _draw_profiles(ax_u1, ax_u2, ax_yc, series_list: list, *, diff: bool,
                    steady: Optional[SteadyRef], percent: bool, yc_cm: bool = True):
    multi = len(series_list) > 1
    x_norm = series_list[0].x_grid / series_list[0].tb.wp.D
    yc_scale = 1.0 if diff else (100.0 if yc_cm else 1.0)  # yc_xt is stored in metres
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    u1_lines, u2_lines, yc_lines = [], [], []
    u1_refs, u2_refs, yc_refs = [], [], []
    for i, s in enumerate(series_list):
        color = colors[i % len(colors)]
        label = f'Turbine {i+1}' if multi else ('Unsteady' if steady is not None else None)
        u1_line, = ax_u1.plot(x_norm, s.u1_xt[0], lw=1.2, color=color)
        u2_line, = ax_u2.plot(x_norm, s.u2_xt[0], lw=1.2, color=color)
        yc_line, = ax_yc.plot(x_norm, s.yc_xt[0] * yc_scale, lw=1.2, color=color, label=label)
        u1_lines.append(u1_line)
        u2_lines.append(u2_line)
        yc_lines.append(yc_line)
        u1_refs.append(s.u1_xt)
        u2_refs.append(s.u2_xt)
        yc_refs.append(s.yc_xt * yc_scale)

    if steady is not None:
        ax_u1.plot(x_norm, steady.u1_x, 'r--', lw=1.0)
        ax_u2.plot(x_norm, steady.u2_x, 'r--', lw=1.0)
        ax_yc.plot(x_norm, steady.yc_x * yc_scale, 'r--', lw=1.0, label=steady.label)
        u1_refs.append(steady.u1_x)
        u2_refs.append(steady.u2_x)
        yc_refs.append(steady.yc_x * yc_scale)

    if multi or steady is not None:
        ax_yc.legend()

    _pad_ylim(ax_u1, *u1_refs, diff=diff)
    _pad_ylim(ax_u2, *u2_refs, diff=diff)
    _pad_ylim(ax_yc, *yc_refs, diff=diff)

    if percent:
        ax_u1.set_ylabel(r'$\Delta u_1/\delta u_{1,\mathrm{env}}$')
        ax_u2.set_ylabel(r'$\Delta u_2/\delta u_{2,\mathrm{env}}$')
        ax_yc.set_ylabel(r'$\Delta y_c/\delta y_{c,\mathrm{env}}$')
    else:
        ax_u1.set_ylabel(r'$\Delta u_1$' if diff else r'$\delta u_1$ [m/s]')
        ax_u2.set_ylabel(r'$\Delta u_2$' if diff else r'$\delta u_2$ [m/s]')
        ax_yc.set_ylabel(r'$\Delta y_c$' if diff else (r'$y_c$ [cm]' if yc_cm else r'$y_c$ [m]'))
    ax_yc.set_xlabel(r'$x/D$')

    if percent:
        for ax in (ax_u1, ax_u2, ax_yc):
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    return u1_lines, u2_lines, yc_lines, yc_scale


def field_video(series: SeriesLike, filename: str, *, steady: Optional[SteadyRef] = None,
                 diff: bool = False, percent: bool = False, gridlines: bool = True,
                 label_fn: Optional[Callable[[int], str]] = None,
                 figsize=(10, 4), **save_kwargs):
    """Animate just the 2D flow field with the centerline(s) overlaid.

    series may be a single WakeSeries or a list (e.g. one per turbine); with a list,
    the field is the sum of the individual fields, with one centreline drawn per series.
    label_fn(i) -> str overrides the per-frame annotation; default is built from each
    series' Turbine yaw via yaw_label_fn.
    """
    series_list = _as_list(series)
    _check_shared_grid(series_list)
    assert all(s.frames is not None and s.y_grid is not None for s in series_list), \
        "series has no field frames -- build one with with_field(series, y_grid) first"
    label_fn = _resolve_label_fn(series_list, label_fn)

    fig, ax = plt.subplots(figsize=figsize)
    mesh, cl_lines, frames, timestamp = _draw_field(ax, fig, series_list, diff=diff, percent=percent, steady=steady)
    ax.set_xlabel(r'$x/D$')
    if gridlines:
        _add_gridlines([ax])

    def update(i):
        mesh.set_array(frames[i].ravel())
        for cl_line, s in zip(cl_lines, series_list):
            cl_line.set_ydata(s.yc_xt[i] / s.tb.wp.D)
        timestamp.set_text(label_fn(i))
        return (mesh, *cl_lines, timestamp)

    ani = FuncAnimation(fig, update, frames=len(series_list[0].ts), interval=50, blit=True)
    save_video(ani, filename, **save_kwargs)
    plt.close(fig)


def profiles_video(series: SeriesLike, filename: str, *, steady: Optional[SteadyRef] = None,
                    diff: bool = False, percent: bool = False, gridlines: bool = True,
                    yc_cm: bool = True, label_fn: Optional[Callable[[int], str]] = None,
                    figsize=(10, 12), **save_kwargs):
    """Animate the three flow-variable profiles (u1, u2, yc) stacked vertically.

    series may be a single WakeSeries or a list (e.g. one per turbine), overlaid
    one line per series. yc_cm: display yc in cm (default) or metres if False.
    label_fn(i) -> str overrides the per-frame annotation; default is built from
    each series' Turbine yaw via yaw_label_fn.
    """
    series_list = _as_list(series)
    _check_shared_grid(series_list)
    label_fn = _resolve_label_fn(series_list, label_fn)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, layout='constrained')
    ax_u1, ax_u2, ax_yc = axes
    u1_lines, u2_lines, yc_lines, yc_scale = _draw_profiles(ax_u1, ax_u2, ax_yc, series_list,
                                                             diff=diff, steady=steady, percent=percent,
                                                             yc_cm=yc_cm)
    if gridlines:
        _add_gridlines(axes)

    timestamp = ax_u1.text(0.02, 0.9, '', transform=ax_u1.transAxes)

    def update(i):
        for u1_line, u2_line, yc_line, s in zip(u1_lines, u2_lines, yc_lines, series_list):
            u1_line.set_ydata(s.u1_xt[i])
            u2_line.set_ydata(s.u2_xt[i])
            yc_line.set_ydata(s.yc_xt[i] * yc_scale)
        timestamp.set_text(label_fn(i))
        return (*u1_lines, *u2_lines, *yc_lines, timestamp)

    ani = FuncAnimation(fig, update, frames=len(series_list[0].ts), interval=50, blit=True)
    save_video(ani, filename, **save_kwargs)
    plt.close(fig)


def full_video(series: SeriesLike, filename: str, *, steady: Optional[SteadyRef] = None,
               diff: bool = False, percent: bool = False, gridlines: bool = True,
               yc_cm: bool = True, label_fn: Optional[Callable[[int], str]] = None,
               figsize=(10, 12), **save_kwargs):
    """Animate the flow field and all three variable profiles together.

    series may be a single WakeSeries or a list (e.g. one per turbine); see
    field_video/profiles_video for how a list is combined in each panel.
    yc_cm: display yc profile in cm (default) or metres if False.
    label_fn(i) -> str overrides the per-frame annotation; default is built from
    each series' Turbine yaw via yaw_label_fn.
    """
    series_list = _as_list(series)
    _check_shared_grid(series_list)
    assert all(s.frames is not None and s.y_grid is not None for s in series_list), \
        "series has no field frames -- build one with with_field(series, y_grid) first"
    label_fn = _resolve_label_fn(series_list, label_fn)

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True, layout='constrained',
                              gridspec_kw={'height_ratios': [2.5, 1, 1, 1]})
    ax_field, ax_u1, ax_u2, ax_yc = axes

    mesh, cl_lines, frames, timestamp = _draw_field(ax_field, fig, series_list, diff=diff, percent=percent, steady=steady,
                                                      colorbar_loc='top', colorbar_shrink=0.5)
    u1_lines, u2_lines, yc_lines, yc_scale = _draw_profiles(ax_u1, ax_u2, ax_yc, series_list,
                                                             diff=diff, steady=steady, percent=percent,
                                                             yc_cm=yc_cm)
    if gridlines:
        _add_gridlines(axes)

    def update(i):
        mesh.set_array(frames[i].ravel())
        for cl_line, s in zip(cl_lines, series_list):
            cl_line.set_ydata(s.yc_xt[i] / s.tb.wp.D)
        for u1_line, u2_line, yc_line, s in zip(u1_lines, u2_lines, yc_lines, series_list):
            u1_line.set_ydata(s.u1_xt[i])
            u2_line.set_ydata(s.u2_xt[i])
            yc_line.set_ydata(s.yc_xt[i] * yc_scale)
        timestamp.set_text(label_fn(i))
        return (mesh, *cl_lines, *u1_lines, *u2_lines, *yc_lines, timestamp)

    ani = FuncAnimation(fig, update, frames=len(series_list[0].ts), interval=50, blit=True)
    save_video(ani, filename, **save_kwargs)
    plt.close(fig)
