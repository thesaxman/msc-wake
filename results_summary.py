"""
Consolidated numeric + static-plot comparison of the Shapiro vs. deficit-advection
wake models, across single-turbine steady/dynamic cases (no yaw / -15 deg / sinusoidal
yaw) and two-turbine combination cases (multiforcing / superposition / sheltered).

Re-solves everything in-process (via the reusable pieces exposed by the other scripts)
rather than parsing their stdout, and writes a summary table plus a handful of static
comparison figures to outputs/ -- no animated video rendering here.
"""

__author__ = "Ali Alebeedan"
__date__ = "5/8/2026"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from model_params import wake_params as wp, solver_params as sp, turbines

import shapiro_steady as ss
import advection_steady as ads
import shapiro_unsteady as su
import deficit_advection_single as das

import multi_forcing as mf_s
import super_position as sp_s
import sheltered as sh_s
import deficit_advection_mf as mf_d
import deficit_advection_sp as sp_d
import deficit_sheltered as sh_d


def max_deficit(u1):
    return float(jnp.max(u1))

def max_defl(yc):
    return float(jnp.max(jnp.abs(yc)))

def max_diff(a, b):
    return float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))


results = {}       # case_name -> (max_deficit [m/s], max_centreline_deflection [m])
comparisons = []   # (label, max_diff_deficit [m/s], max_diff_centreline [m])

# ---------------------------------------------------------------- single-turbine steady
tb0 = turbines[0]  # gamma_deg=-15.0, the only steady case run (advection_steady.py's default)

u1_shap_s = jax.vmap(ss.solve_steady_u1(tb0.wp))(sp.x_grid)
u2_steady_shap = ss.solve_steady_u2(tb0.wp)
yc_shap_s = jax.vmap(ss.solve_steady_yc(tb0.wp, u2_steady_shap))(sp.x_grid)

u1_steady_def = ads.solve_steady_u1(tb0)
u2_steady_def = ads.solve_steady_u2(tb0)
u1_def_s = jax.vmap(u1_steady_def)(sp.x_grid)
yc_def_s = jax.vmap(ads.solve_steady_yc(tb0, u1=u1_steady_def, u2=u2_steady_def))(sp.x_grid)

results["single_steady_shapiro"] = (max_deficit(u1_shap_s), max_defl(yc_shap_s))
results["single_steady_deficit"] = (max_deficit(u1_def_s), max_defl(yc_def_s))
comparisons.append(("single_steady: shapiro vs deficit",
                     max_diff(u1_shap_s, u1_def_s), max_diff(yc_shap_s, yc_def_s)))

# --------------------------------------------------------------- single-turbine dynamic
YAW_CASES = ("no_yaw", "neg15", "sinusoidal")
dyn_shap, dyn_def = {}, {}

for case in YAW_CASES:
    _, x_grid, u1_xt, u2_xt, yc_xt, _ = su.run(case)
    dyn_shap[case] = (u1_xt, yc_xt)
    results[f"single_dynamic_shapiro_{case}"] = (max_deficit(u1_xt), max_defl(yc_xt))

    _, x_grid_d, u1_xt_d, u2_xt_d, yc_xt_d, _ = das.run(case)
    dyn_def[case] = (u1_xt_d, yc_xt_d)
    results[f"single_dynamic_deficit_{case}"] = (max_deficit(u1_xt_d), max_defl(yc_xt_d))

    comparisons.append((f"single_dynamic_{case}: shapiro vs deficit",
                         max_diff(u1_xt, u1_xt_d), max_diff(yc_xt, yc_xt_d)))

# -------------------------------------------------------------------------- two-turbine
METHODS = ("mf", "sp", "sheltered")
two_shap = {"mf": (mf_s.u1_xt, mf_s.yc_xt),
            "sp": (sp_s.u1_xt, sp_s.yc_xt),
            "sheltered": (sh_s.u1_xt, sh_s.yc_xt)}
two_def = {"mf": (mf_d.u1_xt, mf_d.yc_xt),
           "sp": (sp_d.u1_xt, sp_d.yc_xt),
           "sheltered": (sh_d.u1_xt, sh_d.yc_xt)}

for method, (u1, yc) in two_shap.items():
    results[f"two_turbine_shapiro_{method}"] = (max_deficit(u1), max_defl(yc))
for method, (u1, yc) in two_def.items():
    results[f"two_turbine_deficit_{method}"] = (max_deficit(u1), max_defl(yc))

for i, a in enumerate(METHODS):
    for b in METHODS[i + 1:]:
        u1a, yca = two_shap[a]; u1b, ycb = two_shap[b]
        comparisons.append((f"two_turbine_shapiro: {a} vs {b}",
                             max_diff(u1a, u1b), max_diff(yca, ycb)))

for i, a in enumerate(METHODS):
    for b in METHODS[i + 1:]:
        u1a, yca = two_def[a]; u1b, ycb = two_def[b]
        comparisons.append((f"two_turbine_deficit: {a} vs {b}",
                             max_diff(u1a, u1b), max_diff(yca, ycb)))

for m in METHODS:
    u1s, ycs = two_shap[m]; u1d, ycd = two_def[m]
    comparisons.append((f"two_turbine_{m}: shapiro vs deficit",
                         max_diff(u1s, u1d), max_diff(ycs, ycd)))

# ------------------------------------------------------------------------ write summary
lines = []
lines.append("Case results (max deficit = max u1 [m/s], max centreline deflection = max |yc| [m])")
lines.append("-" * 90)
for name, (md, mdefl) in results.items():
    lines.append(f"{name:38s}  max_deficit={md:10.4f}  max_centreline_deflection={mdefl:10.4f}")

lines.append("")
lines.append("Comparisons (max |difference| between the two arrays)")
lines.append("-" * 90)
for label, dmd, dmdefl in comparisons:
    lines.append(f"{label:42s}  max_diff_deficit={dmd:10.4f}  max_diff_centreline={dmdefl:10.4f}")

report = "\n".join(lines)
print(report)
with open("outputs/results_summary.txt", "w", encoding="utf-8") as f:
    f.write(report + "\n")

# ------------------------------------------------------------------------ static plots

# B: single-turbine dynamic, Shapiro vs deficit, final-time profiles, one row per yaw case
fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout="constrained")
for row, case in enumerate(YAW_CASES):
    u1_s, yc_s = dyn_shap[case]
    u1_d, yc_d = dyn_def[case]
    x_norm = x_grid / wp.D
    axes[row, 0].plot(x_norm, u1_s[-1], label="Shapiro", lw=1.5)
    axes[row, 0].plot(x_norm, u1_d[-1], "--", label="Deficit-advection", lw=1.5)
    axes[row, 0].set_ylabel(r"$\delta u_1$ [m/s]")
    axes[row, 0].set_title(f"{case}: final-time deficit")
    axes[row, 0].legend()

    axes[row, 1].plot(x_norm, yc_s[-1], label="Shapiro", lw=1.5)
    axes[row, 1].plot(x_norm, yc_d[-1], "--", label="Deficit-advection", lw=1.5)
    axes[row, 1].set_ylabel(r"$y_c$ [m]")
    axes[row, 1].set_title(f"{case}: final-time centreline")
    axes[row, 1].legend()
axes[-1, 0].set_xlabel(r"$x/D$")
axes[-1, 1].set_xlabel(r"$x/D$")
fig.savefig("outputs/single_turbine_dynamic_comparison.png", dpi=150)
plt.close(fig)

# C: two-turbine, within-model method comparison (mf/sp/sheltered), one column per model
x_norm2 = sp.x_grid / wp.D
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
for m in METHODS:
    u1s, ycs = two_shap[m]
    axes[0, 0].plot(x_norm2, u1s[-1], label=m, lw=1.5)
    axes[1, 0].plot(x_norm2, ycs[-1], label=m, lw=1.5)
    u1d, ycd = two_def[m]
    axes[0, 1].plot(x_norm2, u1d[-1], label=m, lw=1.5)
    axes[1, 1].plot(x_norm2, ycd[-1], label=m, lw=1.5)
axes[0, 0].set_title("Shapiro"); axes[0, 1].set_title("Deficit-advection")
axes[0, 0].set_ylabel(r"$\delta u_1$ [m/s]"); axes[1, 0].set_ylabel(r"$y_c$ [m]")
for ax in axes.ravel():
    ax.set_xlabel(r"$x/D$"); ax.legend()
fig.savefig("outputs/two_turbine_method_comparison.png", dpi=150)
plt.close(fig)

# D: two-turbine, cross-model comparison per method, one row per method
fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout="constrained")
for row, m in enumerate(METHODS):
    u1s, ycs = two_shap[m]
    u1d, ycd = two_def[m]
    axes[row, 0].plot(x_norm2, u1s[-1], label="Shapiro", lw=1.5)
    axes[row, 0].plot(x_norm2, u1d[-1], "--", label="Deficit-advection", lw=1.5)
    axes[row, 0].set_ylabel(r"$\delta u_1$ [m/s]")
    axes[row, 0].set_title(f"{m}: final-time deficit")
    axes[row, 0].legend()

    axes[row, 1].plot(x_norm2, ycs[-1], label="Shapiro", lw=1.5)
    axes[row, 1].plot(x_norm2, ycd[-1], "--", label="Deficit-advection", lw=1.5)
    axes[row, 1].set_ylabel(r"$y_c$ [m]")
    axes[row, 1].set_title(f"{m}: final-time centreline")
    axes[row, 1].legend()
axes[-1, 0].set_xlabel(r"$x/D$")
axes[-1, 1].set_xlabel(r"$x/D$")
fig.savefig("outputs/two_turbine_model_comparison.png", dpi=150)
plt.close(fig)

print("\nWrote outputs/results_summary.txt, outputs/single_turbine_dynamic_comparison.png, "
      "outputs/two_turbine_method_comparison.png, outputs/two_turbine_model_comparison.png")
