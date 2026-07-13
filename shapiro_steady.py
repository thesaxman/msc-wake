"""
As a variant to stead_state.py, this file imports the same model parameters, but uses a different approach to solve the ODEs for the wake centreline deflection and velocity deficit. The results are plotted in a similar manner to steady_state.py.
"""

__author__ = "Ali Alebeedan"
__date__ = "2/7/2026"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, Tsit5, ODETerm, SaveAt, PIDController

from model_params import params
from wake_dynamics import A, dA_dx, G


def solve_steady_u1(p):
    
    gamma_rad = jnp.radians(p.gamma_deg)
    cos_gamma = jnp.cos(gamma_rad)
    S1 = p.UINF*(1-jnp.sqrt(1-p.Ct*cos_gamma**2))
    def du1_dx(x, u1, args): # ODE for u1 definition
        return -dA_dx(x, p)/A(x, p)*u1 + S1*G(x, p)

    u1_sol = diffeqsolve(ODETerm(du1_dx), Tsit5(), t0=p.upstream_bound, t1=p.boundary, dt0=0.01, y0=0.0, saveat=SaveAt(dense=True), stepsize_controller=PIDController(rtol=1e-5, atol=1e-7))

    return u1_sol.evaluate

def solve_steady_u2(p):
    gamma_rad = jnp.radians(p.gamma_deg)
    cos_gamma = jnp.cos(gamma_rad)
    sin_gamma = jnp.sin(gamma_rad)

    def du2_dx(x, u2, args): # ODE for u2 definition
        return -dA_dx(x, p)/A(x, p)*u2 + \
            p.UINF*(1.0/4.0*p.Ct*cos_gamma**2*sin_gamma) \
            *G(x,p)

    u2_sol = diffeqsolve(ODETerm(du2_dx), Tsit5(), t0=p.upstream_bound, t1=p.boundary, dt0=0.01, y0=0.0, saveat=SaveAt(dense=True), stepsize_controller=PIDController(rtol=1e-5, atol=1e-5))

    return u2_sol.evaluate

def solve_steady_yc(p, u2 = None):
    
    if u2 is None:
        u2 = solve_steady_u2(p)
    # This function defines the ODE for the wake centreline deflection
    def dyc_dx(x, y, args):
        return -u2(x) / p.UINF # taking streamwise aligned coordinate
    
    yc_sol = diffeqsolve(ODETerm(dyc_dx), Tsit5(), t0=p.upstream_bound, t1=p.boundary, dt0=0.01, y0=0., saveat=SaveAt(dense=True), stepsize_controller=PIDController(rtol=1e-5, atol=1e-5))

    return yc_sol.evaluate



if __name__ == "__main__":

    #Flow field can be first evaluated along x
    x = jnp.linspace(0.0, params.boundary, 100)
    y = jnp.linspace(-1.1*params.D, 1.1*params.D, 100)
    u1_x = jax.vmap(solve_steady_u1(params))(x)
    u2 = solve_steady_u2(params)
    u2_x = jax.vmap(u2)(x)
    yc_x = jax.vmap(solve_steady_yc(params, u2=u2))(x)
    X, Y = jnp.meshgrid(x, y)
    
    from wake_dynamics import u_point, dw

    u1_field = jax.vmap(u_point, in_axes=(0,0,0,None,None), out_axes=1)(x, yc_x, u1_x, y,params)
    #u2_field = jax.vmap(u_point, in_axes = (0,0,0,None,None), out_axes = 1)(x, yc_x, u2_x, y, params)
    
    # closed form solution for comparison
    
    gamma_rad = jnp.radians(params.gamma_deg)
    cos_gamma = jnp.cos(gamma_rad)
    sin_gamma = jnp.sin(gamma_rad)
    
    u1_0 = params.UINF*(1-jnp.sqrt(1-params.Ct*cos_gamma**2))
    u1_closed = u1_0/(dw(x,params)**2)*0.5*(1+ jax.scipy.special.erf(x/(params.D/2*jnp.sqrt(2))))
    
    u2_0 = params.UINF*(0.25 * params.Ct * cos_gamma**2 * sin_gamma)
    def u2_closed(x):
        return u2_0/(dw(x,params)**2)*0.5*(1+ jax.scipy.special.erf(x/(params.D/2*jnp.sqrt(2))))
    yc_closed = jax.vmap(solve_steady_yc(params, u2=u2_closed))(x)

    u1_closed_field = jax.vmap(u_point, in_axes=(0,0,0,None,None), out_axes=1)(x, yc_closed, u1_closed, y, params)
    #u2_closed_field = jax.vmap(u_point, in_axes=(0,0,0,None,None), out_axes=1)(x, yc_closed, u2_closed(x), y, params)
    
    
    # --- plot ---
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True, layout = 'constrained')
    
    
    # streamwise deficit field
    u1_norm = u1_field/params.UINF
    cf = axes[0].pcolormesh(X/params.D, Y/params.D, u1_norm, cmap='RdBu_r', shading='auto')
    axes[0].plot(x/params.D, yc_x/params.D, color='white', linestyle='dashed', linewidth=1.5,
                label=r'Wake centreline $y_c(x)$')
    plt.colorbar(cf,  ax=axes[0], location='top', shrink=0.5, label=r'$u/U_\infty$ — streamwise deficit')
    axes[0].set_ylabel(r'$y/D$')
    axes[0].set_title(r'Streamwise velocity deficit ($\gamma$ = {:.1f}°)'.format(params.gamma_deg))
    axes[0].legend()
    
    # Comparing deficits from solver to closed form solutions
    u1_closed_norm = u1_closed_field/params.UINF
    residual1 = u1_norm - u1_closed_norm
    m1 = float(jnp.max(jnp.abs(residual1)))
    cf_diff1 = axes[1].pcolormesh(X/params.D, Y/params.D, residual1, cmap='RdBu_r', shading='auto', vmin = -m1, vmax = m1)
    plt.colorbar(cf_diff1, ax=axes[1], location='top', shrink=0.5, label=r'$\Delta u/U_\infty$ — vs closed form')
    axes[1].set_ylabel(r'$y/D$')
    axes[1].set_title('Difference in streamwise velocity deficit to closed form solution')
    print(f"max |Δu|/U∞ = {m1:.2e}")
    
    u2_norm = u2_x/params.UINF
    u2_closed_norm = u2_closed(x)/params.UINF
    residual2 = u2_norm - u2_closed_norm
    m2 = float(jnp.max(jnp.abs(residual2)))
    axes[2].plot(x/params.D, residual2, color = 'red', linewidth=1.5, label = r'$\Delta v/U_infty$ - vs closed form')
    axes[2].set_ylabel(r'$\Delta v/U_\infty$')
    axes[2].set_title('Difference in spanwise velocity deficit to closed form solution')
    print(f"max |Δv|/U∞ = {m2:.2e}")


    # centreline trajectory alone
    axes[3].plot(x/params.D, yc_x/params.D, color='steelblue', linestyle='dashed', linewidth=1.5,
                label=r'Wake centreline $y_c(x)$')
    axes[3].plot(x/params.D, yc_closed/params.D, color = 'black', linestyle = 'solid', linewidth = 1.5, label= r'Closed-form Wake centreline')
    
    axes[3].set_ylabel(r'$y_c/D$')
    axes[3].set_title(r'Wake centreline deflection')
    axes[3].legend()    
    
    
    #Comparing the trajectory from solver to closed form solution
    axes[4].plot(x/params.D, jnp.abs(yc_closed-yc_x)/params.D, color = 'red',linewidth = 1.5, label = r'Wake centreline difference $\Delta y_c(x)$' )
    axes[4].set_title('Wake centreline difference')
    axes[4].set_ylabel(r'$\Delta y_c/D$')
    axes[4].legend()
    print(f"max Δyc/D = {float(jnp.max(jnp.abs(yc_closed-yc_x))/params.D):.2e}")
    
    axes[0].set_xlim(0,12)
    axes[-1].set_xlabel(r'$x/D$')
    #plt.savefig("outputs/wake_field_2d.png", dpi=150)
    plt.show()