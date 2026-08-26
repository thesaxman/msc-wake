""" An idea of introducing a sheltered formulation of the forcings"""

__author__ = "Ali Alebeedan"
__date__ = "29/7/2026"

import jax.numpy as jnp

from model_params import wake_params as wp, solver_params as sp, turbines
from unsteady_flow_solver import solver, sheltered_d_dt
from video_utils import WakeSeries, with_field, full_video


def per_turbine(skew: bool = False):
    """Solve each turbine sheltered by the accumulated upstream deficit, upstream to
    downstream. Returns the list of per-turbine (u1_xt, u2_xt, yc_xt) solutions,
    ordered to match `turbines`."""

    solutions = [(jnp.zeros((sp.nt,sp.nx)), jnp.zeros((sp.nt,sp.nx)), jnp.zeros((sp.nt,sp.nx)))]
    y0 = (jnp.zeros(sp.nx), jnp.zeros(sp.nx), jnp.zeros(sp.nx))

    for tb in turbines:
        solutions.append(solver(y0, sheltered_d_dt(tb, sp, solutions=solutions, skew=skew), sp).ys)
    solutions.pop(0)
    return solutions


def run(skew: bool = False):
    """Same as per_turbine, but summed across turbines into the totals (u1_xt, u2_xt, yc_xt)."""

    solutions = per_turbine(skew)
    u1_xt = sum(u1 for u1, _, _ in solutions)
    u2_xt = sum(u2 for _, u2, _ in solutions)
    yc_xt = sum(yc for _, _, yc in solutions)
    return u1_xt, u2_xt, yc_xt


solutions = per_turbine(False)

if __name__ == "__main__":

    y_grid = jnp.linspace(-3*wp.D, 3*wp.D, 100)

    series_list = with_field(
        [WakeSeries(sp.x_grid, sp.ts, u1_xt, u2_xt, yc_xt, tb)
         for tb, (u1_xt, u2_xt, yc_xt) in zip(turbines, solutions)],
        y_grid,
    )

    full_video(series_list, f'{len(turbines)}_turbine_sheltered.mp4')
