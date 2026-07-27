""" A look at solving the steady state ODE with advection velocity which takes into account the deficit"""

__author__ = "Ali Alebeedan"
__date__ = "25/7/2026"

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from diffrax import Tsit5, PIDController, diffeqsolve, ODETerm, SaveAt
from model_params import wake_params as wp
from wake_dynamics import Turbine, make_turbine, G, A, dA_dx, u_point


p = make_turbine(wp, 0.0)

def solve_steady_u1(p: Turbine):
    
    def adv_d_u1_0(u1, p: Turbine):
        return (p.wp.UINF -u1)* (1-jnp.sqrt(1-p.wp.Ct * jnp.cos(p.wp.gamma) ** 2))
    
    def du1_dx(x, u1, args): # ODE for u1 definition
        return -dA_dx(x, p.wp)/A(x, p.wp)*u1 + adv_d_u1_0(u1, p)*G(x, p.wp)
    
    u1_sol = diffeqsolve(ODETerm(du1_dx), Tsit5(),
                         t0=p.wp.upstream_bound, t1=p.wp.boundary, dt0=0.01, y0=0.0,
                         saveat=SaveAt(dense=True),
                         stepsize_controller=PIDController(rtol=1e-5, atol=1e-7)
                         )
    
    return u1_sol.evaluate

def solve_steady_u2(p: Turbine):
    
    u1 = solve_steady_u1(p)
    
    def adv_d_u2_0(x, u2, p: Turbine):
        return (p.wp.UINF -u1(x))* (0.25*p.wp.Ct * jnp.cos(p.wp.gamma)**2 * jnp.sin(p.wp.gamma))
    
    def du2_dx(x, u2, args): # ODE for u1 definition
        return -dA_dx(x, p.wp)/A(x, p.wp)*u2 + adv_d_u2_0(x, u2, p)*G(x, p.wp)
    
    u2_sol = diffeqsolve(ODETerm(du2_dx), Tsit5(),
                         t0=p.wp.upstream_bound, t1=p.wp.boundary, dt0=0.01, y0=0.0,
                         saveat=SaveAt(dense=True),
                         stepsize_controller=PIDController(rtol=1e-5, atol=1e-7)
                         )
    
    return u2_sol.evaluate

def solve_steady_yc(p: Turbine, u1 = None, u2 = None):
    
    if u1 is None:
        u1 = solve_steady_u1(p)
    if u2 is None:
        u2 = solve_steady_u2(p)
    
    # This function defines the ODE for the wake centreline deflection
    def dyc_dx(x, y, args):
        return -u2(x) / (p.wp.UINF - u1(x))
    
    yc_sol = diffeqsolve(ODETerm(dyc_dx), Tsit5(), 
                         t0=p.wp.upstream_bound, t1=p.wp.boundary, dt0=0.01, y0=0.,
                         saveat=SaveAt(dense=True),
                         stepsize_controller=PIDController(rtol=1e-5, atol=1e-5)
                         )
    
    return yc_sol.evaluate



if __name__ == "__main__":
    
    x = jnp.linspace(0.0, 12.0*wp.D, 100)
    y = jnp.linspace(-1.1*wp.D, 1.1*wp.D, 100)
    u1 = solve_steady_u1(p)
    u1_x = jax.vmap(u1)(x)
    u2 = solve_steady_u2(p)
    u2_x = jax.vmap(u2)(x)
    yc_x = jax.vmap(solve_steady_yc(p, u1=u1, u2=u2))(x)
    X, Y = jnp.meshgrid(x, y)
    
    u1_field = jax.vmap(u_point, in_axes=(0,0,0,None,None), out_axes=1)(x, yc_x, u1_x, y, wp)
    
    print(f'max u/U∞ = {float(u1_field.max()/wp.UINF):.2f}')
    
    #Compare with shapiro steady equations
    
    from shapiro_steady import solve_steady_u1 as u1_steady_shap, solve_steady_u2 as u2_steady_shap, solve_steady_yc as yc_steady_shap
    
    u1_x_shap = jax.vmap(u1_steady_shap(wp))(x)
    u2_shap = u2_steady_shap(wp)
    u2_x_shap = jax.vmap(u2_shap)(x)
    yc_x_shap = jax.vmap(yc_steady_shap(wp, u2 = u2_shap))(x)
    u1_field_shap = jax.vmap(u_point, in_axes=(0,0,0,None,None), out_axes=1)(x, yc_x_shap, u1_x_shap, y, wp)
    
    # --- plot ---
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True, layout = 'constrained')
    
   
    
    # streamwise deficit field
    u1_norm = u1_field/wp.UINF
    cf = axes[0].pcolormesh(X/wp.D, Y/wp.D, u1_norm, cmap='RdBu_r', shading='auto')
    axes[0].plot(x/wp.D, yc_x/wp.D, color='white', linestyle='dashed', linewidth=1.5,
                label=r'Wake centreline $y_c(x)$')
    plt.colorbar(cf,  ax=axes[0], location='top', shrink=0.5, label=r'$u/U_\infty$ — streamwise deficit')
    axes[0].set_ylabel(r'$y/D$')
    axes[0].set_title(r'Streamwise velocity deficit ($\gamma$ = {:.1f}°)'.format(wp.gamma_deg))
    axes[0].legend()
    
    # Comparing deficits from advection to shapiro forms
    u1_shap_norm = u1_field_shap/wp.UINF
    residual1 = u1_norm - u1_shap_norm
    m1 = float(jnp.max(jnp.abs(residual1)))
    cf_diff1 = axes[1].pcolormesh(X/wp.D, Y/wp.D, residual1, cmap='RdBu_r', shading='auto', vmin = -m1, vmax = m1)
    plt.colorbar(cf_diff1, ax=axes[1], location='top', shrink=0.5, label=r'$\Delta u/U_\infty$ — vs Shapiro')
    axes[1].set_ylabel(r'$y/D$')
    axes[1].set_title('Difference in u1 advected vs Shapiro')
    print(f"max |Δu|/U∞ = {m1:.2e}")
    
    u2_norm = u2_x/wp.UINF
    u2_shap_norm = u2_x_shap/wp.UINF
    residual2 = u2_norm - u2_shap_norm
    m2 = float(jnp.max(jnp.abs(residual2)))
    axes[2].plot(x/wp.D, residual2, color = 'red', linewidth=1.5, label = r'$\Delta v/U_infty$ - vs Shapiro')
    axes[2].set_ylabel(r'$\Delta v/U_\infty$')
    axes[2].set_title('Difference in u2 advected vs Shapiro')
    print(f"max |Δv|/U∞ = {m2:.2e}")
    
    
    # centreline trajectory alone
    axes[3].plot(x/wp.D, yc_x/wp.D, color='steelblue', linestyle='dashed', linewidth=1.5,
                label=r'Shapiro centreline $y_c(x)$')
    axes[3].plot(x/wp.D, yc_x_shap/wp.D, color = 'black', linestyle = 'solid', linewidth = 1.5, label= r'Advected centreline')
    
    axes[3].set_ylabel(r'$y_c/D$')
    axes[3].set_title(r'Wake centreline deflection')
    axes[3].legend()
    
    
    #Comparing the trajectory from Shapiro to advected
    axes[4].plot(x/wp.D, jnp.abs(yc_x-yc_x_shap)/wp.D, color = 'red',linewidth = 1.5, label = r'Wake centreline difference $\Delta y_c(x)$' )
    axes[4].set_title('Wake centreline difference')
    axes[4].set_ylabel(r'$\Delta y_c/D$')
    axes[4].legend()
    print(f"max Δyc/D = {float(jnp.max(jnp.abs(yc_x-yc_x_shap))/wp.D):.2e}")
    
    axes[0].set_xlim(0,12)
    axes[-1].set_xlabel(r'$x/D$')
    #plt.savefig("outputs/full_wake_comparisons.png", dpi=150)
    plt.show()