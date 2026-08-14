
from collections.abc import Callable
import dataclasses
#from functools import partial


import jax.numpy as jnp
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController

from wake_dynamics import WakeParams, G, expansion, u_point

### Solver Functions ###

def du1_0(t, p: WakeParams):
    gamma  = p.gamma_at(t)
    return p.UINF*(1-jnp.sqrt(1-p.Ct*jnp.cos(gamma)**2))

def S1_default(t, du1, p: WakeParams):
    return p.UINF * du1_0(t, p)

def du2_0(t, p: WakeParams):
    gamma  = p.gamma_at(t)
    return p.UINF*(0.25*p.Ct*jnp.cos(gamma)**2 * jnp.sin(gamma))

def S2_default(t, du1, p: WakeParams):
    return p.UINF * du2_0(t, p)

def d_dx(var, dx):
    """ First order upwind derivative (flow in +x); zero-gradient inflow BC at index 0. """
    dvar_dx = (var - jnp.roll(var, 1))/dx
    return dvar_dx.at[0].set(0.0)

def d_dx_upwind(var, speed, dx):
    back = (var - jnp.roll(var, 1)) / dx # when speed > 0
    fwd = (jnp.roll(var, -1) - var) / dx # when speed < 0
    d = jnp.where(speed > 0, back, fwd)
    return d.at[0].set(0.0)

def skew_at(U_rotor, du2_rotor):
            """Compute the effective skew angle (in radians) at time t based on the upstream deficit and centerline deflection."""
            skew_angle = jnp.arctan2(-(du2_rotor), U_rotor)
            return skew_angle

### Solver Parameters ###

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

### Turbine tools

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

def turbine_index(tb: Turbine, sp: SolverParams):
    """Index of the turbine in the solver's x-grid."""
    return int(abs((tb.x0 - sp.x_grid)).argmin())

### Shapiro methodology

def default_d_dt(turbines: list[Turbine], sp: SolverParams):
    """Couple RHS: all turbines share one flow field,
    their forcing and expansion effects superpose additively."""

    #precompute forcing spatial terms
    G_xs            = [G(sp.x_grid-tb.x0, tb.wp)         for tb in turbines]
    expansion_xs    = [expansion(sp.x_grid-tb.x0, tb.wp) for tb in turbines]
    def rhs(t, state, args):

        du1, du2, yc = state

        #advection terms
        adv_du1 = -turbines[0].wp.UINF * d_dx(du1, sp.dx)
        adv_du2 = -turbines[0].wp.UINF * d_dx(du2, sp.dx)
        adv_yc = -turbines[0].wp.UINF * d_dx(yc, sp.dx)

        #forcing terms
        src_du1 = sum(-tb.wp.UINF * ex * du1 + sp.S1(t, du1, tb.wp) * Gx
                     for tb, ex, Gx in zip (turbines, expansion_xs, G_xs))
        src_du2 = sum(-tb.wp.UINF * ex * du2 + sp.S2(t, du1, tb.wp) * Gx
                     for tb, ex, Gx in zip (turbines, expansion_xs, G_xs))


        ddu1_dt = adv_du1 + src_du1
        ddu2_dt = adv_du2 + src_du2
        dyc_dt = adv_yc - du2 # du2 is coupled to the centerline deflection equation

        return (ddu1_dt, ddu2_dt, dyc_dt)

    return rhs

### Shapiro with advecting deficit-aware velocity

def adv_S1(t, du1, wp: WakeParams):
    return (wp.UINF - du1) * du1_0(t, wp)

def adv_S2(t, du1, wp: WakeParams):
    return (wp.UINF - du1) * du2_0(t, wp)

def advecting_d_dt(turbines: list[Turbine], sp: SolverParams):
    """Couple RHS: all turbines share one flow field,
    their forcing and expansion effects superpose additively."""

    #precompute forcing spatial terms
    G_xs            = [G(sp.x_grid-tb.x0, tb.wp)         for tb in turbines]
    expansion_xs    = [expansion(sp.x_grid-tb.x0, tb.wp) for tb in turbines]
    UINF = turbines[0].wp.UINF
    def rhs(t, state, args):

        du1, du2, yc = state


        #advection terms
        adv_du1 = -(UINF - du1) * d_dx_upwind(du1, (UINF - du1), sp.dx)
        adv_du2 = -(UINF - du1) * d_dx_upwind(du2, (UINF - du1), sp.dx)
        adv_yc = -(UINF - du1) * d_dx_upwind(yc, (UINF - du1), sp.dx)

        #forcing terms
        src_du1 = sum(-(tb.wp.UINF -du1) * ex * du1 + sp.S1(t, du1, tb.wp) * Gx
                     for tb, ex, Gx in zip (turbines, expansion_xs, G_xs))
        src_du2 = sum(-(tb.wp.UINF - du1) * ex * du2 + sp.S2(t, du1, tb.wp) * Gx
                     for tb, ex, Gx in zip (turbines, expansion_xs, G_xs))


        ddu1_dt = adv_du1 + src_du1
        ddu2_dt = adv_du2 + src_du2
        dyc_dt = adv_yc - du2 # du2 is coupled to the centerline deflection equation

        return (ddu1_dt, ddu2_dt, dyc_dt)

    return rhs

### Sheltered formulation
def _frac(t, sp: SolverParams):
    idx = jnp.clip((t - sp.ts[0]) / (sp.ts[1] - sp.ts[0]), 0.0, sp.ts.size - 1.0)
    i   = jnp.clip(jnp.floor(idx).astype(int), 0, sp.ts.size - 2)
    return i, idx - i

def sheltered_d_dt(tb: Turbine, sp: SolverParams, solutions: list[jnp.ndarray], skew: bool = False):
    """Couple RHS: strictly one turbine is sheltered by the upstream deficit of the other turbine, so we need to compute the upstream deficit at the sheltered turbine's location and adjust its forcing accordingly.
    du1_xt0: (nt, nx) upstream area-averaged deficit sampled at ts"""

    du1_per_turbine = [sol[0] for sol in solutions]
    du2_per_turbine = [sol[1] for sol in solutions]
    du2_sum = sum(du2_per_turbine)
    yc_per_turbine = [sol[2] for sol in solutions]

    #precompute forcing spatial terms
    wp              = tb.wp
    G_x             = G(sp.x_grid-tb.x0, wp)
    expansion_x     = expansion(sp.x_grid-tb.x0, wp)
    i_rotor         = turbine_index(tb, sp)

    def upstream_at(t, yc, du1_xt0, yc_xt0):
        """Linear interp of upstream field in time -> (nx,)"""
        i, f = _frac(t, sp)
        du1_interp = ((1.0-f) * du1_xt0[i] + f * du1_xt0[i+1])
        yc_interp = ((1.0-f) * yc_xt0[i] + f * yc_xt0[i+1])
        return u_point(sp.x_grid, yc_interp, du1_interp, yc, t, wp) # returns the upstream deficit profile at time t
    def du2_at(t, du2_xt0):
        """Linear interp of upstream field in time -> (nx,)"""
        i, f = _frac(t, sp)
        return (1.0-f) * du2_xt0[i] + f * du2_xt0[i+1] # this gives the spatial profile of upstream du2 at time t
    def rhs(t, state, args):

        du1, du2, yc = state

        U_up = wp.UINF - sum(upstream_at(t, yc, du1_xt0, yc_xt0) for du1_xt0, yc_xt0 in zip(du1_per_turbine, yc_per_turbine)) # (nx,) array of local advection velocities at time t
        U_rotor = U_up[i_rotor] # scalar - the local advection velocity at the rotor location of the sheltered turbine

        du2_up = du2_at(t, du2_sum) # (nx,) array of local du2 at time t

        if skew:
            du2_rotor = du2_up[i_rotor] # scalar - the local du2 at the rotor location of the sheltered turbine
            skew_angle = skew_at(U_rotor, du2_rotor)
            gamma_effective = wp.gamma_at(t) - skew_angle # Placeholder for actual control law based on skew angle
        else:
            gamma_effective = wp.gamma_at(t)
        cg, sg = jnp.cos(gamma_effective), jnp.sin(gamma_effective)
        S1 = U_up * U_rotor * (1-jnp.sqrt(1-wp.Ct*cg**2))
        S2 = U_up * U_rotor * (0.25*wp.Ct*cg**2 * sg)

        #advection terms
        adv_du1 = -U_up * d_dx_upwind(du1, U_up, sp.dx)
        adv_du2 = -U_up * d_dx_upwind(du2, U_up, sp.dx)
        adv_yc = -U_up * d_dx_upwind(yc, U_up, sp.dx)

        #forcing terms
        src_du1 = -U_up * expansion_x * du1 + S1 * G_x
        src_du2 = -U_up * expansion_x * du2 + S2 * G_x

        ddu1_dt = adv_du1 + src_du1
        ddu2_dt = adv_du2 + src_du2
        dyc_dt = adv_yc - (du2 + du2_up) # du2 is coupled to the centerline deflection equation

        return (ddu1_dt, ddu2_dt, dyc_dt)
    return rhs

def advecting_sheltered_d_dt(tb: Turbine, sp: SolverParams, solutions: list[jnp.ndarray], skew: bool = False):
    """Couple RHS: strictly one turbine is sheltered by the upstream deficit of the other turbine, so we need to compute the upstream deficit at the sheltered turbine's location and adjust its forcing accordingly.
    du1_xt0: (nt, nx) upstream area-averaged deficit sampled at ts"""

    du1_per_turbine = [sol[0] for sol in solutions]
    du2_per_turbine = [sol[1] for sol in solutions]
    du2_sum = sum(du2_per_turbine)
    yc_per_turbine = [sol[2] for sol in solutions]

    #precompute forcing spatial terms
    wp              = tb.wp
    G_x             = G(sp.x_grid-tb.x0, wp)
    expansion_x     = expansion(sp.x_grid-tb.x0, wp)
    i_rotor         = turbine_index(tb, sp)

    def upstream_at(t, yc, du1_xt0, yc_xt0):
        """Linear interp of upstream field in time -> (nx,)"""
        i, f = _frac(t, sp)
        du1_interp = ((1.0-f) * du1_xt0[i] + f * du1_xt0[i+1])
        yc_interp = ((1.0-f) * yc_xt0[i] + f * yc_xt0[i+1])
        return u_point(sp.x_grid, yc_interp, du1_interp, yc, t, wp) # returns the upstream deficit profile at time t
    def du2_at(t, du2_xt0):
        """Linear interp of upstream field in time -> (nx,)"""
        i, f = _frac(t, sp)
        return (1.0-f) * du2_xt0[i] + f * du2_xt0[i+1]  # this gives the spatial profile of upstream du2 at time t

    def rhs(t, state, args):

        du1, du2, yc = state

        U_up = wp.UINF - sum(upstream_at(t, yc, du1_xt0, yc_xt0) for du1_xt0, yc_xt0 in zip(du1_per_turbine, yc_per_turbine)) # (nx,) array of local advection velocities at time t
        U_rotor = U_up[i_rotor] # scalar - the local advection velocity at the rotor location of the sheltered turbine
        U_local = U_up - du1 # (nx,) array of local advection velocities at time t

        du2_up = du2_at(t, du2_sum) # (nx,) array of local du2 at time t

        if skew:
            du2_rotor = du2_up[i_rotor]
            skew_angle = skew_at(U_rotor, du2_rotor)
            gamma_effective = wp.gamma_at(t) - skew_angle # Placeholder for actual control law based on skew angle
        else:
            gamma_effective = wp.gamma_at(t)
        cg, sg = jnp.cos(gamma_effective), jnp.sin(gamma_effective)
        #It may be argued that the value of U_rotor should be matched to U_local however from the momentum theory the value of du1_0 is dependent on the undisturbed freestream velocity. The current choice takes the stance of only applying U_local to the advection term and using U_rotor for the forcing term.
        S1 = U_local * U_rotor * (1-jnp.sqrt(1-wp.Ct*cg**2))
        S2 = U_local * U_rotor * (0.25*wp.Ct*cg**2 * sg)
        
        #advection terms
        adv_du1 = -U_local * d_dx_upwind(du1, U_local, sp.dx)
        adv_du2 = -U_local * d_dx_upwind(du2, U_local, sp.dx)
        adv_yc = -U_local * d_dx_upwind(yc, U_local, sp.dx)
        
        #forcing terms
        src_du1 = -U_local * expansion_x * du1 + S1 * G_x
        src_du2 = -U_local * expansion_x * du2 + S2 * G_x
        
        ddu1_dt = adv_du1 + src_du1
        ddu2_dt = adv_du2 + src_du2
        dyc_dt = adv_yc - (du2+du2_up) # du2 is coupled to the centerline deflection equation
        
        return (ddu1_dt, ddu2_dt, dyc_dt)
    return rhs


def solver(y0, rhs_func, sp: SolverParams):
    return diffeqsolve(ODETerm(rhs_func), Tsit5(),
                        t0=0, t1=sp.ts[-1], dt0=None, y0=y0,
                        saveat=SaveAt(ts=sp.ts),
                        stepsize_controller=PIDController(rtol=sp.rtol,atol=sp.atol, pcoeff=sp.pcoeff, icoeff=sp.icoeff),
                        max_steps=sp.max_steps)
