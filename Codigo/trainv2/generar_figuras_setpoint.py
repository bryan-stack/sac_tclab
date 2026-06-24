"""
generar_figuras_setpoint.py
============================
Genera las dos figuras complementarias de setpoint constante (40.8 °C)
para los controladores PI Clásico y SAC Puro en hardware real.

Salida: ./figuras_articulo/
  - fig_sp_const_sac.png/.pdf
  - fig_sp_const_pi.png/.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUT_DIR = "./figuras_articulo"
os.makedirs(OUT_DIR, exist_ok=True)

C_PI   = "#d62728"      # rojo
C_SAC  = "#1f77b4"      # azul
C_REF  = "black"
LW     = 2.0
ALPHA  = 0.90

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.6,
})


def plot_single_controller(csv_path, color, label, title_suffix, out_basename):
    """Crea figura con dos paneles: temperatura vs referencia y esfuerzo de control."""
    df = pd.read_csv(csv_path)
    t   = df["Time(s)"].values
    ref = df["Reference(C)"].values
    T   = df["Temperature(C)"].values
    Qu  = df["Control_Effort(%)"].values

    # Métricas
    e     = ref - T
    ise   = np.sum(e**2)
    iae   = np.sum(np.abs(e))
    overshoot = max(0.0, (T - ref).max())
    energy    = np.sum(Qu)
    tv        = np.sum(np.abs(np.diff(Qu)))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.5, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
    )

    # ── Panel superior: temperatura ──────────────────────────────────────────
    ax1.plot(t, ref, color=C_REF, lw=1.8, ls="--",
             label=f"Referencia $T^* = {ref[0]:.2f}$°C")
    ax1.plot(t, T,   color=color, lw=LW, alpha=ALPHA, label=label)

    # Banda de ±0.5 °C alrededor del setpoint (criterio de estado estacionario)
    ax1.fill_between(t, ref - 0.5, ref + 0.5, color=color, alpha=0.06,
                     label="Banda ±0.5°C")

    # Anotar tiempo de subida (al alcanzar el 95% de la diferencia inicial)
    rise_target = ref[0] - 0.05 * (ref[0] - T[0])
    idx_rise    = np.argmax(T >= rise_target) if (T >= rise_target).any() else len(T) - 1
    if idx_rise > 0:
        ax1.axvline(t[idx_rise], color=color, ls=":", alpha=0.5, lw=1.2)
        ax1.text(t[idx_rise] + 5, ref[0] - 4,
                 f"$t_{{95\\%}}={t[idx_rise]:.0f}$s",
                 color=color, fontsize=9.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white",
                           ec=color, alpha=0.85))

    ax1.set_ylabel("Temperatura (°C)")
    ax1.set_title(f"Respuesta en Hardware Real — {title_suffix}  ($T^* = {ref[0]:.2f}$°C)",
                  fontweight="bold")
    ax1.legend(loc="lower right", framealpha=0.92, ncol=1)

    # Texto con métricas
    metrics_txt = (
        f"ISE = {ise:.1f}\n"
        f"IAE = {iae:.1f}\n"
        f"OS$_{{máx}}$ = {overshoot:.2f}°C\n"
        f"$\\int Q_u\\,dt$ = {energy:.0f} %·s\n"
        f"TV = {tv:.1f}"
    )
    ax1.text(0.015, 0.97, metrics_txt, transform=ax1.transAxes,
             va="top", ha="left", fontsize=9.5, family="monospace",
             bbox=dict(boxstyle="round,pad=0.45", fc="#FAFAFA", ec="gray", alpha=0.95))

    ax1.set_ylim(min(T.min(), ref[0]) - 1.5, max(T.max(), ref[0]) + 2.5)

    # ── Panel inferior: esfuerzo de control ──────────────────────────────────
    ax2.step(t, Qu, color=color, lw=1.4, alpha=0.85, where="post",
             label=f"$Q_u$ (%)  |  $\\bar{{Q_u}} = {Qu.mean():.1f}\\%$,  $\\sigma = {Qu.std():.2f}\\%$")
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("$Q_u$ (%)")
    ax2.set_ylim(-2, max(Qu.max() + 5, 102))
    ax2.legend(loc="upper right", framealpha=0.92)

    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))

    plt.savefig(f"{OUT_DIR}/{out_basename}.pdf")
    plt.savefig(f"{OUT_DIR}/{out_basename}.png")
    plt.close(fig)
    print(f"[OK] {out_basename} guardada — ISE={ise:.1f}, OS={overshoot:.2f}°C, TV={tv:.1f}")


# FIGURA: SAC Puro — Setpoint constante 40.8 °C
plot_single_controller(
    csv_path     = "hardware_sac_stage_4.csv",
    color        = C_SAC,
    label        = "Temperatura medida $T(t)$ — SAC Puro",
    title_suffix = "SAC Puro (Stage 4, Curriculum Learning)",
    out_basename = "fig_sp_const_sac",
)

# FIGURA: PI Clásico — Setpoint constante 40.8 °C
plot_single_controller(
    csv_path     = "hardware_pid_comparison.csv",
    color        = C_PI,
    label        = "Temperatura medida $T(t)$ — PI Clásico",
    title_suffix = "PI Clásico (IMC-Skogestad)",
    out_basename = "fig_sp_const_pi",
)

print(f"\nFiguras de setpoint constante guardadas en {OUT_DIR}/")
