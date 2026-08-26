""" Diff videos between the skewed and unskewed variant of a single multi-turbine
    wake model: shapiro (super_position.py), sheltered (sheltered.py) or advected
    deficit (deficit_advection_sp.py). Diffed per turbine -- each turbine's u1/u2/yc
    profiles are normalised against that same turbine's own achieved range (the
    max |u1|/|u2|/|yc| actually reached by either the skewed or unskewed solve),
    not the idealised du1_0/du2_0 forcing-envelope formula: that formula is blind
    to skew itself pushing the *effective* yaw past the raw yaw programme it
    assumes, which silently understates the scale and inflates the normalised
    diff. Select the model via argv[1]:

        python skew_diff.py shapiro
        python skew_diff.py sheltered
        python skew_diff.py advected
"""

import sys

import jax.numpy as jnp

from model_params import solver_params as sp, turbines
from video_utils import WakeSeries, with_field, yaw_label_fn, diff_series, full_video
import super_position as shapiro
import sheltered
import deficit_advection_sp as advected

PER_TURBINE = {
    "shapiro":   shapiro.per_turbine,
    "sheltered": sheltered.per_turbine,
    "advected":  advected.per_turbine,
}


def diff_per_turbine(name: str) -> list:
    """One diff WakeSeries per turbine (skewed - unskewed), each normalised by
    that turbine's own achieved range -- the max |u1|/|u2|/|yc| actually reached
    by either the skewed or unskewed solve."""

    per_turbine = PER_TURBINE[name]
    sols_skew = per_turbine(True)
    sols_unskew = per_turbine(False)

    diffs = []
    for tb, (u1_s, u2_s, yc_s), (u1_u, u2_u, yc_u) in zip(turbines, sols_skew, sols_unskew):
        series_skew = WakeSeries(sp.x_grid, sp.ts, u1_s, u2_s, yc_s, tb)
        series_unskew = WakeSeries(sp.x_grid, sp.ts, u1_u, u2_u, yc_u, tb)

        u1_scale = float(jnp.max(jnp.abs(jnp.stack([series_skew.u1_xt, series_unskew.u1_xt]))))
        u2_scale = float(jnp.max(jnp.abs(jnp.stack([series_skew.u2_xt, series_unskew.u2_xt]))))
        yc_scale = float(jnp.max(jnp.abs(jnp.stack([series_skew.yc_xt, series_unskew.yc_xt]))))

        diffs.append(diff_series(
            series_skew, series_unskew,
            u1_scale=u1_scale, u2_scale=u2_scale, yc_scale=yc_scale,
        ))
    return diffs


if __name__ == "__main__":

    model = sys.argv[1] if len(sys.argv) > 1 else "sheltered"

    print(f"--- {model}: skew vs unskew ---")
    diffs = []
    for i, d in enumerate(diff_per_turbine(model)):
        print(f"turbine {i+1}:")
        diffs.append(d)

    y_grid = jnp.linspace(-3*turbines[0].wp.D, 3*turbines[0].wp.D, 100)
    diffs = with_field(diffs, y_grid)

    full_video(diffs, f'{model}_skew_v_unskew.mp4', diff=True, percent=True,
               label_fn=yaw_label_fn([tb.wp for tb in turbines], sp.ts))
