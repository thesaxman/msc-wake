
from collections.abc import Callable
import dataclasses


import jax.numpy as jnp
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController

from wake_dynamics import WakeParams, G, expansion

def delta_u1_0(t, p: WakeParams):
    gamma  = p.gamma_at(t)
    return p.UINF*(1-jnp.sqrt(1-p.Ct*jnp.cos(gamma)**2))

def S1_default(t, u1, p: WakeParams):
    return p.UINF * delta_u1_0(t, p)
    
def delta_u2_0(t, p: WakeParams):
    gamma  = p.gamma_at(t)
    return p.UINF*(0.25*p.Ct*jnp.cos(gamma)**2 * jnp.sin(gamma))

def S2_default(t, u1, p: WakeParams):
    return p.UINF * delta_u2_0(t, p)

def d_dx(var, dx): 
    """ First order upwind derivative (flow in +x); zero-gradient inflow BC at index 0. """
    dvar_dx = (var - jnp.roll(var, 1))/dx
    return dvar_dx.at[0].set(0.0)

def d_dx_upwind(var, speed, dx):
    back = (var - jnp.roll(var, 1)) / dx # when speed > 0
    fwd = (jnp.roll(var, -1) - var) / dx # when speed < 0
    d = jnp.where(speed > 0, back, fwd)
    return d.at[0].set(0.0)

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
    def tf(self):
        return 2.0 * (self.wp.boundary - self.wp.upstream_bound)/self.wp.UINF
    @property
    def ts(self):
        return jnp.linspace(0, self.tf, self.nt)
    @property
    def x_grid(self):
        return jnp.linspace(self.wp.upstream_bound, self.wp.boundary, self.nx)
    @property
    def dx(self):
        return self.x_grid[1]-self.x_grid[0]

class Turbine(eqx.Module):
    wp: WakeParams
    x0: float = eqx.field(static = True, default = 0.0)

def make_turbine(base: WakeParams, x0_diams: float, *,
                 gamma_deg=None, gamma_fn=None):
    """Build a Turbine from a base WakeParams, overriding only what's given.

    Args:
        base:      the shared baseline WakeParams
        x0_diams:  streamwise position in rotor diameters
        gamma_deg: override base yaw angle (degrees), if given
        gamma_fn:  override control law, if given
    """
    overrides = {}
    if gamma_deg is not None:
        overrides['gamma_deg'] = gamma_deg
    if gamma_fn is not None:
        overrides['gamma_fn'] = gamma_fn
    wp = dataclasses.replace(base, **overrides) if overrides else base
    return Turbine(wp=wp, x0=x0_diams * base.D)

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
        src_u1 = sum(-tb.wp.UINF * ex * u1 + sp.S1(t, u1, tb.wp) * Gx
                     for tb, ex, Gx in zip (turbines, expansion_xs, G_xs))
        src_u2 = sum(-tb.wp.UINF * ex * u2 + sp.S2(t, u1, tb.wp) * Gx
                     for tb, ex, Gx in zip (turbines, expansion_xs, G_xs))
        
        
        du1_dt = adv_u1 + src_u1
        du2_dt = adv_u2 + src_u2
        dyc_dt = adv_yc - u2 # u2 is coupled to the centerline deflection equation
        
        return (du1_dt, du2_dt, dyc_dt)
        
    return rhs

def adv_S1(t, u1, wp: WakeParams):
    return (wp.UINF - u1) * delta_u1_0(t, wp)

def adv_S2(t, u1, wp: WakeParams):
    return (wp.UINF - u1) * delta_u2_0(t, wp)

def advecting_d_dt(turbines: list[Turbine], sp: SolverParams):
    """Couple RHS: all turbines share one flow field,
    their forcing and expansion effects superpose additively."""
    
    #precompute forcing spatial terms
    G_xs            = [G(sp.x_grid-tb.x0, tb.wp)         for tb in turbines]
    expansion_xs    = [expansion(sp.x_grid-tb.x0, tb.wp) for tb in turbines]
    UINF = turbines[0].wp.UINF
    def rhs(t, state, args):
        
        u1, u2, yc = state
        
        
        #advection terms
        adv_u1 = -(UINF - u1) * d_dx_upwind(u1, (UINF - u1), sp.dx)
        adv_u2 = -(UINF - u1) * d_dx_upwind(u2, (UINF - u1), sp.dx)
        adv_yc = -(UINF - u1) * d_dx_upwind(yc, (UINF - u1), sp.dx)
        
        #forcing terms
        src_u1 = sum(-(tb.wp.UINF -u1) * ex * u1 + sp.S1(t, u1, tb.wp) * Gx
                     for tb, ex, Gx in zip (turbines, expansion_xs, G_xs))
        src_u2 = sum(-(tb.wp.UINF - u1) * ex * u2 + sp.S2(t, u1, tb.wp) * Gx
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
                        stepsize_controller=PIDController(rtol=sp.rtol,atol=sp.atol, pcoeff=sp.pcoeff, icoeff=sp.icoeff),
                        max_steps=sp.max_steps)