""" Single-turbine dynamic solve using the deficit-aware advection velocity
    (advects the wake at UINF - u1 rather than pure free-stream), across the
    same yaw scenarios as shapiro_unsteady.py -- keep YAW_CASES in sync. """

__author__ = "Ali Alebeedan"
__date__ = "5/8/2026"

import dataclasses
import sys

import jax
import jax.numpy as jnp
from model_params import wake_params as wp, solver_params as sp

from unsteady_flow_solver import advecting_d_dt, solver, adv_S1, adv_S2
from wake_dynamics import make_turbine, sinusoid_gamma_t
from video_utils import WakeSeries, SteadyRef, with_field, yaw_label_fn, full_video

from advection_steady import solve_steady_u1, solve_steady_u2, solve_steady_yc


YAW_CASES = {
    "no_yaw":     dict(gamma_deg=0.0),
    "neg15":      dict(gamma_deg=-15.0),
    "sinusoidal": dict(gamma_fn=sinusoid_gamma_t),
}

sp_adv = dataclasses.replace(sp, max_steps=10_000_000, S1=adv_S1, S2=adv_S2)


def run(case: str):
    """Solve the single-turbine deficit-advection dynamic wake for a given yaw case.

    Returns (turbine, x_grid, u1_xt, u2_xt, yc_xt, steady_or_None); the steady
    reference is skipped for the sinusoidal case, as in shapiro_unsteady.run.
    """
    turbine = make_turbine(wp, 0.0, **YAW_CASES[case])

    x_grid = sp_adv.x_grid
    y0 = (jnp.zeros(sp_adv.nx), jnp.zeros(sp_adv.nx), jnp.zeros(sp_adv.nx))

    u1_xt, u2_xt, yc_xt = solver(y0=y0, rhs_func=advecting_d_dt([turbine], sp_adv), sp=sp_adv).ys

    steady = None
    if case != "sinusoidal":
        u1_steady = solve_steady_u1(turbine)
        u2_steady = solve_steady_u2(turbine)
        yc_steady = solve_steady_yc(turbine, u1=u1_steady, u2=u2_steady)
        steady = SteadyRef(jax.vmap(u1_steady)(x_grid),
                            jax.vmap(u2_steady)(x_grid),
                            jax.vmap(yc_steady)(x_grid))

    return turbine, x_grid, u1_xt, u2_xt, yc_xt, steady


if __name__ == "__main__":

    case = sys.argv[1] if len(sys.argv) > 1 else "no_yaw"
    turbine, x_grid, u1_xt, u2_xt, yc_xt, steady = run(case)

    print(float((wp.UINF - u1_xt).min()))

    y_grid = jnp.linspace(-3.0*wp.D, 3*wp.D, 100)

    series = with_field(
        WakeSeries(x_grid, sp_adv.ts, u1_xt, u2_xt, yc_xt, wp,
                   label_fn=yaw_label_fn([turbine.wp], sp_adv.ts)),
        y_grid,
    )

    full_video(series, f'deficit_unsteady_{case}.mp4', steady=steady)
