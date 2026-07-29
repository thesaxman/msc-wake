""" A simple approach to visualising effect of n turbines in a single column"""

__author__ = "Ali Alebeedan"
__date__ = "16/7/2026"

import jax.numpy as jnp
from model_params import wake_params, solver_params as sp
from wake_dynamics import make_turbine, sinusoid_gamma_t
from unsteady_flow_solver import default_d_dt, solver

from video_utils import WakeSeries, with_field, yaw_label_fn, full_video


turbine1 = make_turbine(wake_params, 0.0, gamma_deg=-15.0) # dynamic yaw, upstream
turbine2 = make_turbine(wake_params, 5.0, gamma_fn=sinusoid_gamma_t) # static baseline, downstream

y0 = (jnp.zeros(sp.nx), jnp.zeros(sp.nx), jnp.zeros(sp.nx))

u1_xt1, u2_xt1, yc_xt1 = solver(y0, default_d_dt([turbine1], sp), sp).ys
u1_xt2, u2_xt2, yc_xt2 = solver(y0, default_d_dt([turbine2], sp), sp).ys
u1_xt = u1_xt1 + u1_xt2
u2_xt = u2_xt1 + u2_xt2
yc_xt = yc_xt1 + yc_xt2

if __name__ == "__main__":

    y_grid = jnp.linspace(-3*wake_params.D, 3*wake_params.D, 100)

    series = with_field(
        WakeSeries(sp.x_grid, sp.ts, u1_xt, u2_xt, yc_xt, wake_params,
                   label_fn=yaw_label_fn([turbine1.wp], sp.ts)),
        y_grid,
    )

    full_video(series, 'two_turbine_superposition.mp4')
