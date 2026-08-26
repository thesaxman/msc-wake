""" Diff videos between pairs of the three multi-turbine wake models: shapiro
    (super_position.py), sheltered (sheltered.py) and advected deficit
    (deficit_advection_sp.py). Diffed per turbine -- each turbine's u1/u2/yc
    profiles are normalised against that same turbine's own achieved range (the
    max |u1|/|u2|/|yc| actually reached by either of the two solves being
    compared), not a single shared scale, since the turbines carry very
    different yaw programmes. This is deliberately empirical rather than the
    idealised du1_0/du2_0 forcing-envelope formula: that formula is blind to
    anything (like skew, in skew_diff.py) that pushes the *effective* yaw past
    the raw yaw programme it assumes, which silently understates the scale and
    inflates the normalised diff. Select the pair via argv[1]:

        python multi_turbine_diff.py sheltered_v_shapiro
        python multi_turbine_diff.py shapiro_v_advected
        python multi_turbine_diff.py sheltered_v_advected
"""

import sys

import jax.numpy as jnp

from model_params import solver_params as sp, turbines
from video_utils import WakeSeries, with_field, yaw_label_fn, diff_series, full_video
import super_position as shapiro
import sheltered
import deficit_advection_sp as advected

PAIRS = {
    "sheltered_v_shapiro":  ("sheltered", "shapiro"),
    "shapiro_v_advected":   ("shapiro", "advected"),
    "sheltered_v_advected": ("sheltered", "advected"),
}

PER_TURBINE = {
    "shapiro":   shapiro.per_turbine,
    "sheltered": sheltered.per_turbine,
    "advected":  advected.per_turbine,
}


def diff_per_turbine(name_a: str, name_b: str) -> list:
    """One diff WakeSeries per turbine, each normalised by that turbine's own
    achieved range -- the max |u1|/|u2|/|yc| actually reached by either of the
    two solves being compared."""

    sols_a = PER_TURBINE[name_a](False)
    sols_b = PER_TURBINE[name_b](False)

    diffs = []
    for tb, (u1_a, u2_a, yc_a), (u1_b, u2_b, yc_b) in zip(turbines, sols_a, sols_b):
        series_a = WakeSeries(sp.x_grid, sp.ts, u1_a, u2_a, yc_a, tb)
        series_b = WakeSeries(sp.x_grid, sp.ts, u1_b, u2_b, yc_b, tb)

        u1_scale = float(jnp.max(jnp.abs(jnp.stack([series_a.u1_xt, series_b.u1_xt]))))
        u2_scale = float(jnp.max(jnp.abs(jnp.stack([series_a.u2_xt, series_b.u2_xt]))))
        yc_scale = float(jnp.max(jnp.abs(jnp.stack([series_a.yc_xt, series_b.yc_xt]))))

        diffs.append(diff_series(
            series_a, series_b,
            u1_scale=u1_scale, u2_scale=u2_scale, yc_scale=yc_scale,
        ))
    return diffs


if __name__ == "__main__":

    pair = sys.argv[1] if len(sys.argv) > 1 else "sheltered_v_advected"
    name_a, name_b = PAIRS[pair]

    print(f"--- {pair} ---")
    diffs = []
    for i, d in enumerate(diff_per_turbine(name_a, name_b)):
        print(f"turbine {i+1}:")
        diffs.append(d)

    y_grid = jnp.linspace(-3*turbines[0].wp.D, 3*turbines[0].wp.D, 100)
    diffs = with_field(diffs, y_grid)

    full_video(diffs, f'{pair}_diff.mp4', diff=True, percent=True,
               label_fn=yaw_label_fn([tb.wp for tb in turbines], sp.ts))
