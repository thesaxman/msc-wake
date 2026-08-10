"""_summary_ This script is an attempt at adding time-dependence in forcing.
"""

__author__ = "Ali Alebeedan"
__date__ = "10/7/2026"

import jax
import jax.numpy as jnp
from model_params import wake_params as wp, solver_params as sp
from wake_dynamics import sinusoid_gamma_t
from unsteady_flow_solver import default_d_dt, solver, make_turbine
from video_utils import WakeSeries, with_field, full_video


if __name__ == "__main__":
    import shapiro_steady as ss
    u1_x = jax.vmap(ss.solve_steady_u1(wp))(sp.x_grid)
    u2_steady = ss.solve_steady_u2(wp)
    u2_x = jax.vmap(u2_steady)(sp.x_grid)
    yc_x = jax.vmap(ss.solve_steady_yc(wp, u2_steady))(sp.x_grid)
    y0 = (u1_x, u2_x, yc_x)

    turbine = make_turbine(wp, 0.0, gamma_fn=sinusoid_gamma_t)

    u1_xt, u2_xt, yc_xt = solver(y0, default_d_dt([turbine], sp), sp).ys
    y_grid = jnp.linspace(-3*turbine.wp.D, 3*turbine.wp.D, 100)

    series = with_field(
        WakeSeries(sp.x_grid, sp.ts, u1_xt, u2_xt, yc_xt, turbine),
        y_grid,
    )

    full_video(series, 'sinusoidal_yaw.mp4')
