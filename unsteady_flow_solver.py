####### Still figuring this out... ########

from collections.abc import Callable


import jax.numpy as jnp
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController

from wake_dynamics import WakeParams, G, expansion

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

def d_dx(var, dx): 
    """ First order upwind derivative (flow in +x); zero-gradient inflow BC at index 0. """
    dvar_dx = (var - jnp.roll(var, 1))/dx
    return dvar_dx.at[0].set(0.0)


class SolverParams(eqx.Module):
    
    wp:         WakeParams
    nx:         int = eqx.field(static=True, default=800)
    nt:         int = eqx.field(static=True, default=1000)
    S1:         Callable = eqx.field(static = True, default = S1_default)
    S2:         Callable = eqx.field(static = True, default = S2_default)
    max_steps:  int = eqx.field(static=True, default=1_000_000)
    rtol:       float = 1e-5
    atol:       float = 1e-7
    pcoeff:     float = 0.3
    icoeff:     float = 0.3
    
    
    @property
    def t1(self):
        return 2.0 * (self.wp.boundary - self.wp.upstream_bound)/self.wp.UINF
    @property
    def ts(self):
        return jnp.linspace(0, self.t1, self.nt)
    @property
    def x_grid(self):
        return jnp.linspace(self.wp.upstream_bound, self.wp.boundary, self.nx)
    @property
    def dx(self):
        return self.x_grid[1]-self.x_grid[0]

def default_d_dt(wp: WakeParams, sp: SolverParams, x_offset: float = 0.0):
    # Single turbine wake field - Can also be used for superposition implementation
    G_x = G(sp.x_grid-x_offset, wp)
    expansion_x = expansion(sp.x_grid-x_offset, wp)
    def rhs(t, state, args):
        
        u1, u2, yc = state
        du1_dt = -wp.UINF * d_dx(u1, sp.dx) - wp.UINF * expansion_x * u1 \
            + sp.S1(t, wp) * G_x
        du2_dt = -wp.UINF * d_dx(u2, sp.dx) - wp.UINF * expansion_x * u2 \
            + sp.S2(t, wp) * G_x
        
        dyc_dt = -wp.UINF * d_dx(yc, sp.dx) - u2 # u2 is coupled to the centerline deflection equation
        
        return (du1_dt, du2_dt, dyc_dt)
        
    return rhs

def solver(y0, rhs_func, sp: SolverParams):
    return diffeqsolve(ODETerm(rhs_func), Tsit5(),
                        t0=0, t1=sp.ts[-1], dt0=None, y0=y0,
                        saveat=SaveAt(ts=sp.ts),
                        stepsize_controller=PIDController(rtol=sp.rtol,atol=sp.atol),
                        max_steps=sp.max_steps)