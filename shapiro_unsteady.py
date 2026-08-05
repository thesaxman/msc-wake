"""
A try at implementing the unsteady PDE to see if Diffrax can take it.
"""

__author__ = "Ali Alebeedan"
__date__ = "2/7/2026"

import sys

import jax
import jax.numpy as jnp
from model_params import wake_params as wp, solver_params as sp

import unsteady_flow_solver as flowsolve
import shapiro_steady as ss
from wake_dynamics import make_turbine, sinusoid_gamma_t
from video_utils import WakeSeries, SteadyRef, with_field, yaw_label_fn, field_video, profiles_video, full_video


# Single-turbine yaw scenarios shared with deficit_advection_single.py -- keep in sync.
YAW_CASES = {
    "no_yaw":     dict(gamma_deg=0.0),
    "neg15":      dict(gamma_deg=-15.0),
    "sinusoidal": dict(gamma_fn=sinusoid_gamma_t),
}


def run(case: str):
    """Solve the single-turbine Shapiro-model dynamic wake for a given yaw case.

    Returns (turbine, x_grid, u1_xt, u2_xt, yc_xt, steady_or_None), where `steady`
    is a SteadyRef built from the matching closed-form steady solution (skipped for
    the sinusoidal case, which has no valid static reference).
    """
    turbine = make_turbine(wp, 0.0, **YAW_CASES[case])

    x_grid = sp.x_grid
    y0 = (jnp.zeros(sp.nx),
          jnp.zeros(sp.nx),
          jnp.zeros(sp.nx))

    u1_xt, u2_xt, yc_xt = flowsolve.solver(y0, flowsolve.default_d_dt([turbine], sp), sp).ys

    steady = None
    if case != "sinusoidal":
        u1_x = jax.vmap(ss.solve_steady_u1(turbine.wp))(x_grid)
        u2_steady = ss.solve_steady_u2(turbine.wp)
        u2_x = jax.vmap(u2_steady)(x_grid)
        yc_x = jax.vmap(ss.solve_steady_yc(turbine.wp, u2_steady))(x_grid)
        steady = SteadyRef(u1_x, u2_x, yc_x)

    return turbine, x_grid, u1_xt, u2_xt, yc_xt, steady


if __name__ == "__main__":

    case = sys.argv[1] if len(sys.argv) > 1 else "no_yaw"
    turbine, x_grid, u1_xt, u2_xt, yc_xt, steady = run(case)

    y_grid = jnp.linspace(-3*wp.D, 3*wp.D, 100)

    series = with_field(
        WakeSeries(x_grid, sp.ts, u1_xt, u2_xt, yc_xt, wp, label_fn=yaw_label_fn([turbine.wp], sp.ts)),
        y_grid,
    )

    full_video(series, f'shapiro_unsteady_{case}.mp4', steady=steady)
