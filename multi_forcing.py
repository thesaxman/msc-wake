""" A simple approach to visualising effect of n turbines in a single column"""

__author__ = "Ali Alebeedan"
__date__ = "16/7/2026"

from functools import partial

import jax.numpy as jnp
from model_params import wake_params as wp, solver_params as sp
from wake_dynamics import sinusoid_gamma_t, make_turbine
from unsteady_flow_solver import default_d_dt, solver

from video_utils import WakeSeries, with_field, yaw_label_fn, full_video


mk = partial(make_turbine, wp)

turbines = [
    mk(0.0),
    mk(5.0, gamma_fn=sinusoid_gamma_t),
    mk(10.0, gamma_deg=-15.0)
]

y0 = (jnp.zeros(sp.nx), jnp.zeros(sp.nx), jnp.zeros(sp.nx))

u1_xt, u2_xt, yc_xt = solver(y0=y0, rhs_func=default_d_dt(turbines, sp), sp=sp).ys

if __name__ == "__main__":
    D = wp.D
    x_grid = sp.x_grid
    y_grid = jnp.linspace(-3.0*D, 3*D, 100)

    series = with_field(
        WakeSeries(x_grid, sp.ts, u1_xt, u2_xt, yc_xt, wp,
                   label_fn=yaw_label_fn([tb.wp for tb in turbines], sp.ts)),
        y_grid,
    )

    full_video(series, 'three_turbine_mf.mp4')
