"""
The following script organises the model parameters used for calculating the
flow field. The parameters are collected as wake dynamics parameters and solver scheme parameters.
"""
from functools import partial

from wake_dynamics import WakeParams, make_turbine, sinusoid_gamma_t
from unsteady_flow_solver import SolverParams

# This is from Bastankhah and Porté-Agel wind-tunnel setup with data fitted for kw and sigma0
BASTANKHAH = WakeParams(
    D=0.15, kw=0.0834, Ct=0.8, gamma_deg=0.0, UINF=4.88,
)

NREL5MW = WakeParams(
    D=126, kw = 0.31, Ct=0.8, gamma_deg=0.0, UINF=11.4, calibration="bastankhah-2021" # sigma not fitted so just for illustrative purposes 
)

MODELS = {
    "bastankhah":    BASTANKHAH,
    "nrel5mw":       NREL5MW,
}

ACTIVE = "bastankhah"

wake_params = MODELS[ACTIVE]
solver_params = SolverParams(wp=wake_params)

mk = partial(make_turbine, wake_params)

turbine1 = mk(0.0, gamma_deg=-15.0)
turbine2 = mk(5.0, gamma_fn=sinusoid_gamma_t)
#turbine3 = mk(10.0, gamm)

turbines = [turbine1, turbine2]