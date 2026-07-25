
from collections.abc import Callable


import jax.numpy as jnp
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController

from wake_dynamics import WakeParams, G, expansion, Turbine

def delta_u1_0(t, p: WakeParams):
    gamma  = p.gamma_at(t)
    return p.UINF*(1-jnp.sqrt(1-p.Ct*jnp.cos(gamma)**2))

def S1_default(t, p: WakeParams):
    return p.UINF * delta_u1_0(t, p)
    
def delta_u2_0(t, p: WakeParams):
    gamma  = p.gamma_at(t)
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


def default_d_dt(turbines: list[Turbine], sp: SolverParams):
    """Couple RHS: all turbines share one flow field,
    their forcing and expansion effects superpose additively."""
    
    #precompute forcing spatial terms
    G_xs            = [G(sp.x_grid-tb.x0, tb.wp)         for tb in turbines]
    expansion_xs    = [expansion(sp.x_grid-tb.x0, tb.wp) for tb in turbines]
    def rhs(t, state, args):
        
        u1, u2, yc = state
        
        #advection terms
        adv_u1 = -turbines[0].wp.UINF * d_dx(u1, sp.dx)
        adv_u2 = -turbines[0].wp.UINF * d_dx(u2, sp.dx)
        adv_yc = -turbines[0].wp.UINF * d_dx(yc, sp.dx)
        
        #forcing terms
        src_u1 = sum(-tb.wp.UINF * ex * u1 + sp.S1(t, tb.wp) * Gx
                     for tb, ex, Gx in zip (turbines, expansion_xs, G_xs))
        src_u2 = sum(-tb.wp.UINF * ex * u2 + sp.S2(t, tb.wp) * Gx
                     for tb, ex, Gx in zip (turbines, expansion_xs, G_xs))
        
        
        du1_dt = adv_u1 + src_u1
        du2_dt = adv_u2 + src_u2
        dyc_dt = adv_yc - u2 # u2 is coupled to the centerline deflection equation
        
        return (du1_dt, du2_dt, dyc_dt)
        
    return rhs

def advecting_d_dt(turbines: list[Turbine], sp: SolverParams):
    """Couple RHS: all turbines share one flow field,
    their forcing and expansion effects superpose additively."""
    
    #precompute forcing spatial terms
    G_xs            = [G(sp.x_grid-tb.x0, tb.wp)         for tb in turbines]
    expansion_xs    = [expansion(sp.x_grid-tb.x0, tb.wp) for tb in turbines]
    def rhs(t, state, args):
        
        u1, u2, yc = state
        
        
        #advection terms
        adv_u1 = -(turbines[0].wp.UINF - u1) * d_dx(u1, sp.dx)
        adv_u2 = -(turbines[0].wp.UINF - u1) * d_dx(u2, sp.dx)
        adv_yc = -(turbines[0].wp.UINF - u1) * d_dx(yc, sp.dx)
        
        #forcing terms
        src_u1 = sum(-(tb.wp.UINF -u1) * ex * u1 + sp.S1(t, tb.wp) * Gx
                     for tb, ex, Gx in zip (turbines, expansion_xs, G_xs))
        src_u2 = sum(-(tb.wp.UINF - u1) * (1-u1/tb.wp.UINF) * ex * u2 + sp.S2(t, tb.wp) * Gx
                     for tb, ex, Gx in zip (turbines, expansion_xs, G_xs))
        
        
        du1_dt = adv_u1 + src_u1
        du2_dt = adv_u2 + src_u2
        dyc_dt = adv_yc - u2 # u2 is coupled to the centerline deflection equation
        
        return (du1_dt, du2_dt, dyc_dt)
        
    return rhs

def solver(y0, rhs_func, sp: SolverParams):
    return diffeqsolve(ODETerm(rhs_func), Tsit5(),
                        t0=0, t1=sp.ts[-1], dt0=None, y0=y0,
                        saveat=SaveAt(ts=sp.ts),
                        stepsize_controller=PIDController(rtol=sp.rtol,atol=sp.atol),
                        max_steps=sp.max_steps)