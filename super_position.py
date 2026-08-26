""" A simple approach to visualising effect of n turbines in a single column"""

__author__ = "Ali Alebeedan"
__date__ = "16/7/2026"

import jax.numpy as jnp
from model_params import wake_params as wp, solver_params as sp, turbines
from unsteady_flow_solver import default_d_dt, solver

from video_utils import WakeSeries, with_field, yaw_label_fn, full_video


def per_turbine(skew: bool = False):
    """Solve each turbine's wake independently (superposed additively), upstream to
    downstream, optionally correcting each turbine's effective yaw for the skew
    induced by the upstream turbines' wakes at its own rotor. Returns the list of
    per-turbine (u1_xt, u2_xt, yc_xt) solutions, ordered to match `turbines`."""

    y0 = (jnp.zeros(sp.nx), jnp.zeros(sp.nx), jnp.zeros(sp.nx))
    solutions = [(jnp.zeros((sp.nt, sp.nx)), jnp.zeros((sp.nt, sp.nx)), jnp.zeros((sp.nt, sp.nx)))]
    for tb in turbines:
        solutions.append(solver(y0, default_d_dt([tb], sp, solutions=solutions, skew=skew), sp).ys)
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

if __name__ == "__main__":

    y_grid = jnp.linspace(-3*wp.D, 3*wp.D, 100)

    series = with_field(
        WakeSeries(sp.x_grid, sp.ts, u1_xt, u2_xt, yc_xt, turbines[0]),
        y_grid,
    )

    full_video(series, f'{len(turbines)}_turbine_superposition.mp4',
               label_fn=yaw_label_fn([tb.wp for tb in turbines], sp.ts))
