""" An idea of introducing a sheltered formulation of the forcings"""

__author__ = "Ali Alebeedan"
__date__ = "29/7/2026"

import dataclasses

import jax.numpy as jnp
from model_params import wake_params as wp, solver_params as sp, turbines
from unsteady_flow_solver import default_d_dt, solver, sheltered_d_dt

from video_utils import WakeSeries, with_field, yaw_label_fn, full_video


solutions = []
y0 = (jnp.zeros(sp.nx), jnp.zeros(sp.nx), jnp.zeros(sp.nx))

u1_xt1, u2_xt1, yc_xt1 = solver(y0, default_d_dt([turbines[0]], sp), sp).ys

solutions.append((u1_xt1, u2_xt1, yc_xt1))

if len(turbines) > 1:
    for tb in turbines[1:]:
        solutions.append(solver(y0, sheltered_d_dt(tb, sp, solutions),sp).ys)

if __name__ == "__main__":

    y_grid = jnp.linspace(-3*wp.D, 3*wp.D, 100)

    series = with_field(
        WakeSeries(sp.x_grid, sp.ts, u1_xt, u2_xt, yc_xt, wp,
                   label_fn=yaw_label_fn([tb.wp for tb in turbines], sp.ts)),
        y_grid,
    )

    full_video(series, 'two_turbine_sheltered.mp4')
