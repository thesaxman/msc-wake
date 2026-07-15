####### Still figuring this out... ########

from collections.abc import Callable

import jax
import jax.numpy as jnp
import equinox as eqx

from wake_dynamics import WakeParams
from model_params import wake_params

def delta_u1_0(t, p: WakeParams):
     return p.UINF*(1-jnp.sqrt(1-p.Ct*jnp.cos(p.gamma_fn(t, p.sigma0))))

def S1_default(t, p: WakeParams):
    return p.UINF * delta_u1_0(t, p)
    
def delta_u2_0(t, p: WakeParams):
     return p.UINF*(0.25-jnp.sqrt(1-p.Ct*jnp.cos(p.gamma_fn(t, p.sigma0))))

def S2_default(t, p: WakeParams):
    return p.UINF * delta_u1_0(t, p)

class solver_params(eqx.Module):
    
    nx:     float
    S1:     Callable = eqx.field(static = True, default = S1_default)
    S2:     Callable = eqx.field(static = True, default = S2_default)