""" Two-turbine superposition solve using the deficit-aware advection velocity
    (advects the wake at UINF - u1 rather than purely free-stream) -- each turbine
    solved independently then summed, the deficit-advection counterpart of
    super_position.py."""

__author__ = "Ali Alebeedan"
__date__ = "25/7/2026"

import dataclasses

import jax.numpy as jnp

from model_params import wake_params as wp, solver_params as sp, turbines
from unsteady_flow_solver import advecting_d_dt, solver, adv_S1, adv_S2
from video_utils import WakeSeries, with_field, yaw_label_fn, full_video


sp = dataclasses.replace(sp, max_steps  = 10_000_000, S1 = adv_S1, S2 = adv_S2)

y0 = (jnp.zeros(sp.nx), jnp.zeros(sp.nx), jnp.zeros(sp.nx))

u1_xt1, u2_xt1, yc_xt1 = solver(y0=y0, rhs_func=advecting_d_dt([turbines[0]], sp), sp=sp).ys
u1_xt2, u2_xt2, yc_xt2 = solver(y0=y0, rhs_func=advecting_d_dt([turbines[1]], sp), sp=sp).ys
u1_xt = u1_xt1 + u1_xt2
u2_xt = u2_xt1 + u2_xt2
yc_xt = yc_xt1 + yc_xt2

print(float((wp.UINF - u1_xt).min()))

if __name__ == "__main__":

    D = wp.D
    x_grid = sp.x_grid
    y_grid = jnp.linspace(-3.0*D, 3*D, 100)

    series = with_field(
        WakeSeries(x_grid, sp.ts, u1_xt, u2_xt, yc_xt, turbines[0]),
        y_grid,
    )
    full_video(series, 'two_turbine_deficit_superposition.mp4',
               label_fn=yaw_label_fn([tb.wp for tb in turbines], sp.ts))
