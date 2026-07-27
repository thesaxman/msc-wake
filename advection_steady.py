""" A look at solving the steady state ODE with advection velocity which takes into account the deficit"""

__author__ = "Ali Alebeedan"
__date__ = "25/7/2026"

from functools import partial
import dataclasses

import jax.numpy as jnp
from model_params import wake_params as wp, solver_params as sp
from wake_dynamics import Turbine, make_turbine

from video_utils import WakeSeries, with_field, yaw_label_fn, full_video

make_turbine(wp, 0.0)

def solve_steady_u1(p: Turbine):
    
    def du1_dx(x, u1, args): # ODE for u1 definition
        return -dA_dx(x, p)/A(x, p)*u1 + S1_default([],p)*G(x, p)/p.UINF
    
    u1_sol = diffeqsolve(ODETerm(du1_dx), Tsit5(),
                         t0=p.upstream_bound, t1=p.boundary, dt0=0.01, y0=0.0,
                         saveat=SaveAt(dense=True),
                         stepsize_controller=PIDController(rtol=1e-5, atol=1e-7)
                         )