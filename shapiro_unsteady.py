"""
A try at implementing the unsteady PDE to see if Diffrax can take it.
"""

__author__ = "Ali Alebeedan"
__date__ = "2/7/2026"

import jax
import jax.numpy as jnp
from model_params import wake_params as wp, solver_params as sp

import unsteady_flow_solver as flowsolve
import shapiro_steady as ss
from wake_dynamics import make_turbine
from video_utils import WakeSeries, SteadyRef, with_field, yaw_label_fn, field_video, profiles_video, full_video


if __name__ == "__main__":

    x_grid = sp.x_grid
    turbine = make_turbine(wp, 0.0)

    y0 = (jnp.zeros(sp.nx),
          jnp.zeros(sp.nx),
          jnp.zeros(sp.nx))

    u1_xt, u2_xt, yc_xt = flowsolve.solver(y0, flowsolve.default_d_dt([turbine], sp), sp).ys

    y_grid = jnp.linspace(-3*wp.D, 3*wp.D, 100)
    u1_x = jax.vmap(ss.solve_steady_u1(wp))(x_grid)
    u2_steady = ss.solve_steady_u2(wp)
    u2_x = jax.vmap(u2_steady)(x_grid)
    yc_x = jax.vmap(ss.solve_steady_yc(wp, u2_steady))(x_grid)

    series = with_field(
        WakeSeries(x_grid, sp.ts, u1_xt, u2_xt, yc_xt, wp, label_fn=yaw_label_fn([wp], sp.ts)),
        y_grid,
    )
    steady = SteadyRef(u1_x, u2_x, yc_x)

    #field_video(series, 'wake_field_evolution.mp4', steady=steady)
    #profiles_video(series, 'wake_profiles_evolution.mp4', steady=steady)
    #full_video(series, 'full_wake_evolution.mp4', steady=steady)
