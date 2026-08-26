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


def per_turbine(skew: bool = False):
    """Solve each turbine's wake independently with deficit-aware advection
    (superposed additively), upstream to downstream, optionally correcting each
    turbine's effective yaw for the skew induced by the upstream turbines' wakes
    at its own rotor. Returns the list of per-turbine (u1_xt, u2_xt, yc_xt)
    solutions, ordered to match `turbines`."""

    y0 = (jnp.zeros(sp.nx), jnp.zeros(sp.nx), jnp.zeros(sp.nx))
    solutions = [(jnp.zeros((sp.nt, sp.nx)), jnp.zeros((sp.nt, sp.nx)), jnp.zeros((sp.nt, sp.nx)))]
    for tb in turbines:
        solutions.append(solver(y0, advecting_d_dt([tb], sp, solutions=solutions, skew=skew), sp).ys)
    solutions.pop(0)
    return solutions


def run(skew: bool = False):
    """Same as per_turbine, but summed across turbines into the totals (u1_xt, u2_xt, yc_xt)."""

    solutions = per_turbine(skew)
    u1_xt = sum(u1 for u1, _, _ in solutions)
    u2_xt = sum(u2 for _, u2, _ in solutions)
    yc_xt = sum(yc for _, _, yc in solutions)
    return u1_xt, u2_xt, yc_xt


u1_xt, u2_xt, yc_xt = run(False)

print(float((wp.UINF - u1_xt).min()))

if __name__ == "__main__":

    D = wp.D
    x_grid = sp.x_grid
    y_grid = jnp.linspace(-3.0*D, 3*D, 100)

    series = with_field(
        WakeSeries(x_grid, sp.ts, u1_xt, u2_xt, yc_xt, turbines[0]),
        y_grid,
    )
    full_video(series, f'{len(turbines)}_turbine_deficit_superposition.mp4',
               label_fn=yaw_label_fn([tb.wp for tb in turbines], sp.ts))
