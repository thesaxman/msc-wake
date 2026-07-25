"""
Functional video-building library for wake-model variables.

Collects the animation boilerplate repeated across shapiro_unsteady.py,
multi_forcing.py, super_position.py, varying_turbine.py and
multi_turbine_diff.py into three reusable views built on a common
WakeSeries data container:

    field_video(series, filename)      -- just the 2D flow field + centerline
    profiles_video(series, filename)   -- just the u1 / u2 / yc profiles
    full_video(series, filename)       -- field and profiles together

Pass diff=True (with a series built via diff_series) to render a
difference between two runs, e.g. multi-forcing vs. superposition.
"""

import os
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator, PercentFormatter

from wake_dynamics import u_point, WakeParams

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def make_field_frames(x_grid, yc_xt, u1_xt, y_grid, wp):
    build_frame = jax.vmap(u_point, in_axes=(0, 0, 0, None, None), out_axes=1)
    all_frames = jax.vmap(build_frame, in_axes=(None, 0, 0, None, None), out_axes=0)(x_grid, yc_xt, u1_xt, y_grid, wp)
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
    wp:       WakeParams
    frames:   Optional[np.ndarray] = None   # (nt, ny, nx) -- required by field_video/full_video
    y_grid:   Optional[np.ndarray] = None   # paired with frames
    label_fn: Optional[Callable[[int], str]] = None  # per-frame annotation, defaults to "t = .. s"

    def __post_init__(self):
        self.x_grid = np.asarray(self.x_grid)
        self.ts = np.asarray(self.ts)
        self.u1_xt = np.asarray(self.u1_xt)
        self.u2_xt = np.asarray(self.u2_xt)
        self.yc_xt = np.asarray(self.yc_xt)


@dataclass
class SteadyRef:
    """A time-invariant reference solution overlaid as a dashed red line."""
    u1_x: np.ndarray
    u2_x: np.ndarray
    yc_x: np.ndarray
    label: str = "Steady solution"


def with_field(series: WakeSeries, y_grid) -> WakeSeries:
    """Return a copy of series with flow-field frames computed over y_grid."""
    frames = make_field_frames(series.x_grid, series.yc_xt, series.u1_xt, y_grid, series.wp)
    return dataclasses.replace(series, frames=frames, y_grid=np.asarray(y_grid))


def diff_series(a: WakeSeries, b: WakeSeries, *, u1_scale=1.0, u2_scale=1.0, yc_scale=1.0,
                 label_fn=None) -> WakeSeries:
    """Elementwise a - b, optionally normalised per-variable (e.g. by envelope magnitudes)."""
    assert np.allclose(a.x_grid, b.x_grid), "series must share the same x grid"
    assert np.allclose(a.ts, b.ts), "series must share the same time grid"
    return WakeSeries(
        x_grid=a.x_grid, ts=a.ts, wp=a.wp, label_fn=label_fn,
        u1_xt=(np.asarray(a.u1_xt) - np.asarray(b.u1_xt)) / u1_scale,
        u2_xt=(np.asarray(a.u2_xt) - np.asarray(b.u2_xt)) / u2_scale,
        yc_xt=(np.asarray(a.yc_xt) - np.asarray(b.yc_xt)) / yc_scale,
    )


def _label(series: WakeSeries, i: int) -> str:
    if series.label_fn is not None:
        return series.label_fn(i)
    return rf"t = {series.ts[i]:.0f} s"


def yaw_label_fn(params, ts) -> Callable[[int], str]:
    """Per-frame label showing yaw angle(s) alongside time, for use as WakeSeries.label_fn.

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
        yaw_str = ", ".join(rf"$\gamma_{{{j}}}$={y:.1f}°" for j, y in enumerate(yaws))
        return rf"{yaw_str}, t = {t:.0f} s"
    return label


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


def _draw_field(ax, fig, series: WakeSeries, *, diff: bool, percent: bool,
                 steady: Optional[SteadyRef], colorbar_loc='right', colorbar_shrink=None):
    wp = series.wp
    x_norm = series.x_grid / wp.D
    y_norm = series.y_grid / wp.D
    frames = series.frames

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
        cl_label = r'$\Delta y_c$'
    else:
        mesh = ax.pcolormesh(x_norm, y_norm, frames[0], cmap='plasma', shading='auto', vmin=0, vmax=frames.max())
        cbar_kwargs['label'] = r'$u_1$ [m/s]'
        cl_label = 'Unsteady $y_c$' if steady is not None else '$y_c$'

    fig.colorbar(mesh, ax=ax, **cbar_kwargs)
    cl_line, = ax.plot(x_norm, series.yc_xt[0] / wp.D, 'w--', lw=1.2, label=cl_label)

    if steady is not None:
        ax.plot(x_norm, steady.yc_x / wp.D, 'r--', lw=1.0, label=steady.label)

    ax.set_ylabel(r'$y/D$')
    ax.legend(loc='upper right')
    timestamp = ax.text(0.02, 0.92, '', transform=ax.transAxes, color='w')
    return mesh, cl_line, timestamp


def _draw_profiles(ax_u1, ax_u2, ax_yc, series: WakeSeries, *, diff: bool,
                    steady: Optional[SteadyRef], percent: bool):
    x_norm = series.x_grid / series.wp.D

    u1_line, = ax_u1.plot(x_norm, series.u1_xt[0], lw=1.2)
    u2_line, = ax_u2.plot(x_norm, series.u2_xt[0], lw=1.2)
    yc_line, = ax_yc.plot(x_norm, series.yc_xt[0], lw=1.2, label='Unsteady' if steady is not None else None)

    u1_refs, u2_refs, yc_refs = [series.u1_xt], [series.u2_xt], [series.yc_xt]
    if steady is not None:
        ax_u1.plot(x_norm, steady.u1_x, 'r--', lw=1.0)
        ax_u2.plot(x_norm, steady.u2_x, 'r--', lw=1.0)
        ax_yc.plot(x_norm, steady.yc_x, 'r--', lw=1.0, label=steady.label)
        ax_yc.legend()
        u1_refs.append(steady.u1_x)
        u2_refs.append(steady.u2_x)
        yc_refs.append(steady.yc_x)

    _pad_ylim(ax_u1, *u1_refs, diff=diff)
    _pad_ylim(ax_u2, *u2_refs, diff=diff)
    _pad_ylim(ax_yc, *yc_refs, diff=diff)

    ax_u1.set_ylabel(r'$\Delta u_1$' if diff else r'$u_1$ [m/s]')
    ax_u2.set_ylabel(r'$\Delta u_2$' if diff else r'$u_2$ [m/s]')
    ax_yc.set_ylabel(r'$\Delta y_c$' if diff else r'$y_c$ [m]')
    ax_yc.set_xlabel(r'$x/D$')

    if percent:
        for ax in (ax_u1, ax_u2, ax_yc):
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    return u1_line, u2_line, yc_line


def field_video(series: WakeSeries, filename: str, *, steady: Optional[SteadyRef] = None,
                 diff: bool = False, percent: bool = False, gridlines: bool = True,
                 figsize=(10, 4), **save_kwargs):
    """Animate just the 2D flow field with the centerline overlaid."""
    assert series.frames is not None and series.y_grid is not None, \
        "series has no field frames -- build one with with_field(series, y_grid) first"

    fig, ax = plt.subplots(figsize=figsize)
    mesh, cl_line, timestamp = _draw_field(ax, fig, series, diff=diff, percent=percent, steady=steady)
    ax.set_xlabel(r'$x/D$')
    if gridlines:
        _add_gridlines([ax])

    def update(i):
        mesh.set_array(series.frames[i].ravel())
        cl_line.set_ydata(series.yc_xt[i] / series.wp.D)
        timestamp.set_text(_label(series, i))
        return mesh, cl_line, timestamp

    ani = FuncAnimation(fig, update, frames=len(series.ts), interval=50, blit=True)
    save_video(ani, filename, **save_kwargs)
    plt.close(fig)


def profiles_video(series: WakeSeries, filename: str, *, steady: Optional[SteadyRef] = None,
                    diff: bool = False, percent: bool = False, gridlines: bool = True,
                    figsize=(10, 12), **save_kwargs):
    """Animate the three flow-variable profiles (u1, u2, yc) stacked vertically."""
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, layout='constrained')
    ax_u1, ax_u2, ax_yc = axes
    u1_line, u2_line, yc_line = _draw_profiles(ax_u1, ax_u2, ax_yc, series,
                                                diff=diff, steady=steady, percent=percent)
    if gridlines:
        _add_gridlines(axes)

    timestamp = ax_u1.text(0.02, 0.9, '', transform=ax_u1.transAxes)

    def update(i):
        u1_line.set_ydata(series.u1_xt[i])
        u2_line.set_ydata(series.u2_xt[i])
        yc_line.set_ydata(series.yc_xt[i])
        timestamp.set_text(_label(series, i))
        return u1_line, u2_line, yc_line, timestamp

    ani = FuncAnimation(fig, update, frames=len(series.ts), interval=50, blit=True)
    save_video(ani, filename, **save_kwargs)
    plt.close(fig)


def full_video(series: WakeSeries, filename: str, *, steady: Optional[SteadyRef] = None,
               diff: bool = False, percent: bool = False, gridlines: bool = True,
               figsize=(10, 12), **save_kwargs):
    """Animate the flow field and all three variable profiles together."""
    assert series.frames is not None and series.y_grid is not None, \
        "series has no field frames -- build one with with_field(series, y_grid) first"

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True, layout='constrained',
                              gridspec_kw={'height_ratios': [2.5, 1, 1, 1]})
    ax_field, ax_u1, ax_u2, ax_yc = axes

    mesh, cl_line, timestamp = _draw_field(ax_field, fig, series, diff=diff, percent=percent, steady=steady,
                                            colorbar_loc='top', colorbar_shrink=0.5)
    u1_line, u2_line, yc_line = _draw_profiles(ax_u1, ax_u2, ax_yc, series,
                                                diff=diff, steady=steady, percent=percent)
    if gridlines:
        _add_gridlines(axes)

    def update(i):
        mesh.set_array(series.frames[i].ravel())
        cl_line.set_ydata(series.yc_xt[i] / series.wp.D)
        u1_line.set_ydata(series.u1_xt[i])
        u2_line.set_ydata(series.u2_xt[i])
        yc_line.set_ydata(series.yc_xt[i])
        timestamp.set_text(_label(series, i))
        return mesh, cl_line, u1_line, u2_line, yc_line, timestamp

    ani = FuncAnimation(fig, update, frames=len(series.ts), interval=50, blit=True)
    save_video(ani, filename, **save_kwargs)
    plt.close(fig)
