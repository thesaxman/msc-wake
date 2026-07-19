####### Still figuring this out... ########

from collections.abc import Callable

import jax
import jax.numpy as jnp
import equinox as eqx

from wake_dynamics import WakeParams

def delta_u1_0(t, p: WakeParams):
    gamma  = p.gamma_fn(t, p.gamma)
    return p.UINF*(1-jnp.sqrt(1-p.Ct*jnp.cos(gamma)**2))

def S1_default(t, p: WakeParams):
    return p.UINF * delta_u1_0(t, p)
    
def delta_u2_0(t, p: WakeParams):
    gamma  = p.gamma_fn(t, p.gamma)
    return p.UINF*(0.25*p.Ct*jnp.cos(gamma)**2 * jnp.sin(gamma))

def S2_default(t, p: WakeParams):
    return p.UINF * delta_u2_0(t, p)

class SolverParams(eqx.Module):
    
    nx:     int
    nt:     int
    S1:     Callable = eqx.field(static = True, default = S1_default)
    S2:     Callable = eqx.field(static = True, default = S2_default)