""" Sheltered formulation of the forcings, deficit-advection counterpart of sheltered.py:
    turbine 1 is solved with the deficit-aware advection velocity (advects at UINF - u1),
    then turbine 2 sees a locally-reduced UINF equal to turbine 1's upstream deficit at
    turbine 2's location, also solved with the deficit-aware advection velocity."""

__author__ = "Ali Alebeedan"
__date__ = "5/8/2026"

import dataclasses

import jax.numpy as jnp
from model_params import wake_params as wp, solver_params as sp, turbines
from unsteady_flow_solver import advecting_d_dt, solver, adv_S1, adv_S2

from video_utils import WakeSeries, with_field, yaw_label_fn, full_video


sp_adv = dataclasses.replace(sp, max_steps=10_000_000, S1=adv_S1, S2=adv_S2)

y0 = (jnp.zeros(sp_adv.nx), jnp.zeros(sp_adv.nx), jnp.zeros(sp_adv.nx))

u1_xt1, u2_xt1, yc_xt1 = solver(y0, advecting_d_dt([turbines[0]], sp_adv), sp_adv).ys

u1_xt2, u2_xt2, yc_xt2 = solver(y0, advecting_d_dt([turbines[1]], sp2_adv), sp2_adv).ys
u1_xt = u1_xt1 + u1_xt2
u2_xt = u2_xt1 + u2_xt2
yc_xt = yc_xt1 + yc_xt2

if __name__ == "__main__":

    y_grid = jnp.linspace(-3*wp.D, 3*wp.D, 100)

    series = with_field(
        WakeSeries(sp_adv.x_grid, sp_adv.ts, u1_xt, u2_xt, yc_xt, turbines[0]),
        y_grid,
    )

    full_video(series, 'two_turbine_deficit_sheltered.mp4',
               label_fn=yaw_label_fn([tb.wp for tb in turbines], sp_adv.ts))
