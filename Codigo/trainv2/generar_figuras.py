"""
generar_figuras.py  —  Generador de figuras para el artículo MDPI Processes
===========================================================================
Genera todas las figuras auxiliares necesarias para el artículo.
Ejecutar desde: trainv2/
Salida: figuras en ./figuras_articulo/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import FancyArrowPatch

OUT_DIR = "./figuras_articulo"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Paleta de colores consistente con el artículo ───────────────────────────
C_PI   = "#d62728"      # rojo
C_SAC  = "#1f77b4"      # azul
C_REF  = "#2ca02c"      # verde oscuro (referencia)
C_FILL = "#aec7e8"      # azul claro (fondo SAC)
ALPHA  = 0.85
LW     = 2.0

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.6,
})

# ── Carga de datos ────────────────────────────────────────────────────────────
df_pi  = pd.read_csv("stress_data_pi.csv")
df_sac = pd.read_csv("stress_data_sac.csv")

t_pi   = df_pi["Time(s)"].values
T_pi   = df_pi["Temperature(C)"].values
R_pi   = df_pi["Reference(C)"].values
Q_pi   = df_pi["Control_Effort(%)"].values

t_sac  = df_sac["Time(s)"].values
T_sac  = df_sac["Temperature(C)"].values
R_sac  = df_sac["Reference(C)"].values
Q_sac  = df_sac["Control_Effort(%)"].values

e_pi   = R_pi  - T_pi
e_sac  = R_sac - T_sac

# Fases del perfil de estrés
PHASES = [(0, 150, 35), (150, 300, 48), (300, 450, 28), (450, 600, 40)]
PHASE_LABELS = ["Fase 1\n(35°C)", "Fase 2\n(48°C)", "Fase 3\n(28°C)", "Fase 4\n(40°C)"]

def add_phase_bands(ax, alpha=0.05, colors=["#d4e6f1","#fde8d8","#e8f8e8","#fef9e7"]):
    for i, (t0, t1, _) in enumerate(PHASES):
        ax.axvspan(t0, t1, alpha=alpha, color=colors[i], zorder=0)


# FIGURA 1: Comparativa Principal (ya existe como resultado_final_monografia.png)
# Se rehace con mayor calidad tipográfica
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})

add_phase_bands(ax1)
add_phase_bands(ax2)

ax1.plot(t_pi,  R_pi,  "k--", lw=1.8, label="Referencia $T^*(t)$", zorder=5)
ax1.plot(t_pi,  T_pi,  color=C_PI,  lw=LW, alpha=ALPHA, label="PI Clásico (IMC-Skogestad)")
ax1.plot(t_sac, T_sac, color=C_SAC, lw=LW, alpha=ALPHA, label="SAC Puro (Propuesto)")

# Anotar el sobreimpulso máximo del PI
t_os_pi  = t_pi[np.argmin(e_pi)]
T_os_pi  = T_pi[np.argmin(e_pi)]
ax1.annotate(f"OS$_{{PI}}$ = 20.90°C\n($t$={t_os_pi:.0f}s)",
             xy=(t_os_pi, T_os_pi), xytext=(t_os_pi-70, T_os_pi+1.5),
             arrowprops=dict(arrowstyle="->", color=C_PI, lw=1.4),
             fontsize=9.5, color=C_PI,
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_PI, alpha=0.85))

# Anotar el sobreimpulso máximo del SAC
t_os_sac  = t_sac[np.argmin(e_sac)]
T_os_sac  = T_sac[np.argmin(e_sac)]
ax1.annotate(f"OS$_{{SAC}}$ = 13.94°C\n($t$={t_os_sac:.0f}s)",
             xy=(t_os_sac, T_os_sac), xytext=(t_os_sac+20, T_os_sac+3),
             arrowprops=dict(arrowstyle="->", color=C_SAC, lw=1.4),
             fontsize=9.5, color=C_SAC,
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_SAC, alpha=0.85))

# Etiquetas de fase
for (t0, t1, _), lbl in zip(PHASES, PHASE_LABELS):
    ax1.text((t0+t1)/2, 50.8, lbl, ha="center", va="bottom", fontsize=8.5,
             color="gray", style="italic")

ax1.set_ylabel("Temperatura (°C)")
ax1.set_ylim(27, 52)
ax1.legend(loc="upper left", framealpha=0.9)
ax1.set_title("Prueba de Estrés Dinámico en Hardware Real — PI Clásico vs. SAC Puro",
              fontweight="bold")

ax2.step(t_pi,  Q_pi,  color=C_PI,  lw=1.4, alpha=0.75, where="post", label="$Q_u$ PI (%)")
ax2.step(t_sac, Q_sac, color=C_SAC, lw=1.4, alpha=0.80, where="post", label="$Q_u$ SAC (%)")
ax2.set_xlabel("Tiempo (s)")
ax2.set_ylabel("Esfuerzo de\nControl (%)")
ax2.set_ylim(-5, 108)
ax2.legend(loc="upper right", framealpha=0.9)

for ax in (ax1, ax2):
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))

plt.savefig(f"{OUT_DIR}/fig1_comparativa_principal.pdf")
plt.savefig(f"{OUT_DIR}/fig1_comparativa_principal.png")
plt.close()
print("[OK] fig1_comparativa_principal guardada")


# FIGURA 2: Trayectoria de Error e(t) = T_ref - T_real
fig, ax = plt.subplots(figsize=(11, 4))
add_phase_bands(ax)

ax.axhline(0, color="black", lw=1.2, linestyle="--", alpha=0.5, zorder=4)

# Rellenos para sobreimpulso (error < 0) y subimpulso (error > 0)
ax.fill_between(t_pi,  0, e_pi,  where=(e_pi<0),  alpha=0.25, color=C_PI,
                label="Sobreimpulso PI")
ax.fill_between(t_sac, 0, e_sac, where=(e_sac<0), alpha=0.25, color=C_SAC,
                label="Sobreimpulso SAC")

ax.plot(t_pi,  e_pi,  color=C_PI,  lw=LW, alpha=ALPHA, label="$e(t)$ PI Clásico")
ax.plot(t_sac, e_sac, color=C_SAC, lw=LW, alpha=ALPHA, label="$e(t)$ SAC Puro")

# Anotar los mínimos de error
ax.annotate(f"−20.90°C", xy=(t_pi[np.argmin(e_pi)], np.min(e_pi)),
            xytext=(t_pi[np.argmin(e_pi)]-70, -18),
            arrowprops=dict(arrowstyle="->", color=C_PI),
            color=C_PI, fontsize=9.5)
ax.annotate(f"−13.94°C", xy=(t_sac[np.argmin(e_sac)], np.min(e_sac)),
            xytext=(t_sac[np.argmin(e_sac)]+20, -12),
            arrowprops=dict(arrowstyle="->", color=C_SAC),
            color=C_SAC, fontsize=9.5)

ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Error $e(t) = T^* - T$ (°C)")
ax.set_title("Trayectoria del Error de Seguimiento — PI Clásico vs. SAC Puro",
             fontweight="bold")
ax.legend(loc="lower right", ncol=2, framealpha=0.9)
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.xaxis.set_minor_locator(MultipleLocator(10))

plt.savefig(f"{OUT_DIR}/fig2_error_trajectory.pdf")
plt.savefig(f"{OUT_DIR}/fig2_error_trajectory.png")
plt.close()
print("[OK] fig2_error_trajectory guardada")


# FIGURA 3: Zoom — Evento de Windup (t = 100s a 280s)
ZOOM_T0, ZOOM_T1 = 100, 280
mask_pi  = (t_pi  >= ZOOM_T0) & (t_pi  <= ZOOM_T1)
mask_sac = (t_sac >= ZOOM_T0) & (t_sac <= ZOOM_T1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1], "hspace": 0.1})

ax1.axvspan(150, ZOOM_T1, alpha=0.08, color="#fde8d8")
ax1.axvline(150, color="gray", lw=1.2, ls="--", label="Escalón $T^*=48$°C")

ax1.plot(t_pi[mask_pi],  R_pi[mask_pi],  "k--", lw=1.8)
ax1.plot(t_pi[mask_pi],  T_pi[mask_pi],  color=C_PI,  lw=LW, alpha=ALPHA, label="PI Clásico")
ax1.plot(t_sac[mask_sac], T_sac[mask_sac], color=C_SAC, lw=LW, alpha=ALPHA, label="SAC Puro")

# Marcadores de sobreimpulso
t_win = t_pi[mask_pi][np.argmax(T_pi[mask_pi])]
T_win = np.max(T_pi[mask_pi])
ax1.annotate(f"Windup PI\n{T_win:.2f}°C (+{T_win-48:.2f}°C)",
             xy=(t_win, T_win), xytext=(t_win-60, T_win-2),
             arrowprops=dict(arrowstyle="->", color=C_PI, lw=1.3),
             fontsize=9, color=C_PI,
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_PI, alpha=0.85))

ax1.set_ylabel("Temperatura (°C)")
ax1.legend(loc="upper left", framealpha=0.9)
ax1.set_title("Zoom: Evento de Windup Integral (Escalón +13°C, $t=150$s)",
              fontweight="bold")

ax2.axvline(150, color="gray", lw=1.2, ls="--")
ax2.axhline(100, color=C_PI, lw=0.8, ls=":", alpha=0.6)
ax2.step(t_pi[mask_pi],  Q_pi[mask_pi],  color=C_PI,  lw=1.4, where="post", alpha=0.8)
ax2.step(t_sac[mask_sac], Q_sac[mask_sac], color=C_SAC, lw=1.4, where="post", alpha=0.85)
ax2.text(155, 102, "Saturación ($Q_u=100\%$)", fontsize=8.5, color=C_PI)
ax2.set_xlabel("Tiempo (s)")
ax2.set_ylabel("$Q_u$ (%)")
ax2.set_ylim(-5, 110)

plt.savefig(f"{OUT_DIR}/fig3_zoom_windup.pdf")
plt.savefig(f"{OUT_DIR}/fig3_zoom_windup.png")
plt.close()
print("[OK] fig3_zoom_windup guardada")


# FIGURA 4: Zoom — Enfriamiento Asimétrico (t = 270s a 460s)
ZOOM_T0, ZOOM_T1 = 270, 460
mask_pi  = (t_pi  >= ZOOM_T0) & (t_pi  <= ZOOM_T1)
mask_sac = (t_sac >= ZOOM_T0) & (t_sac <= ZOOM_T1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1], "hspace": 0.1})

ax1.axvspan(300, 450, alpha=0.08, color="#e8f8e8")
ax1.axvline(300, color="gray", lw=1.2, ls="--", label="Escalón $T^*=28$°C")
ax1.axhline(28, color="green", lw=1.0, ls=":", alpha=0.7, label="Ref = 28°C")

ax1.plot(t_pi[mask_pi],  R_pi[mask_pi],  "k--", lw=1.8)
ax1.plot(t_pi[mask_pi],  T_pi[mask_pi],  color=C_PI,  lw=LW, alpha=ALPHA, label="PI Clásico")
ax1.plot(t_sac[mask_sac], T_sac[mask_sac], color=C_SAC, lw=LW, alpha=ALPHA, label="SAC Puro")

# Flechas de brechas
ax1.annotate("", xy=(350, 40.23), xytext=(350, 28),
             arrowprops=dict(arrowstyle="<->", color="gray", lw=1.3))
ax1.text(352, 34, "PI: min\n40.23°C\n(Δ=12.23°C)", fontsize=8, color=C_PI)

ax1.annotate("", xy=(380, 37.66), xytext=(380, 28),
             arrowprops=dict(arrowstyle="<->", color=C_SAC, lw=1.3, linestyle="dashed"))
ax1.text(382, 32.5, "SAC: min\n37.66°C\n(Δ=9.66°C)", fontsize=8, color=C_SAC)

ax1.set_ylabel("Temperatura (°C)")
ax1.set_title("Zoom: Régimen de Enfriamiento Pasivo — Asimetría Térmica ($T^*=28$°C)",
              fontweight="bold")
ax1.legend(loc="upper right", framealpha=0.9, ncol=2)

# Control effort (expected ~0 for both)
ax2.axvline(300, color="gray", lw=1.2, ls="--")
ax2.step(t_pi[mask_pi],  Q_pi[mask_pi],  color=C_PI,  lw=1.4, where="post", alpha=0.8)
ax2.step(t_sac[mask_sac], Q_sac[mask_sac], color=C_SAC, lw=1.4, where="post", alpha=0.85)
ax2.set_xlabel("Tiempo (s)")
ax2.set_ylabel("$Q_u$ (%)")
ax2.set_ylim(-5, 30)

plt.savefig(f"{OUT_DIR}/fig4_zoom_cooling.pdf")
plt.savefig(f"{OUT_DIR}/fig4_zoom_cooling.png")
plt.close()
print("[OK] fig4_zoom_cooling guardada")


# FIGURA 5: Análisis de Chattering — Señal de Control en Detalle (t=50..140s)
ZOOM_T0, ZOOM_T1 = 50, 145
mask_pi  = (t_pi  >= ZOOM_T0) & (t_pi  <= ZOOM_T1)
mask_sac = (t_sac >= ZOOM_T0) & (t_sac <= ZOOM_T1)

fig, axes = plt.subplots(2, 2, figsize=(12, 6))
fig.suptitle("Análisis de Chattering de la Señal de Control (Estado Cuasi-Estacionario, $t\\in[50,145]$s)",
             fontweight="bold")

# Señal temporal PI
ax = axes[0, 0]
ax.step(t_pi[mask_pi], Q_pi[mask_pi], color=C_PI, lw=1.2, where="post")
ax.set_title("Señal $Q_u(t)$ — PI Clásico")
ax.set_ylabel("$Q_u$ (%)")
ax.set_xlabel("Tiempo (s)")
q_pi_ss = Q_pi[mask_pi]
ax.text(0.02, 0.95, f"TV={np.sum(np.abs(np.diff(q_pi_ss))):.1f}  σ={q_pi_ss.std():.2f}%",
        transform=ax.transAxes, va="top", fontsize=9.5,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7))

# Señal temporal SAC
ax = axes[0, 1]
ax.step(t_sac[mask_sac], Q_sac[mask_sac], color=C_SAC, lw=1.2, where="post")
ax.set_title("Señal $Q_u(t)$ — SAC Puro")
ax.set_ylabel("$Q_u$ (%)")
ax.set_xlabel("Tiempo (s)")
q_sac_ss = Q_sac[mask_sac]
ax.text(0.02, 0.95, f"TV={np.sum(np.abs(np.diff(q_sac_ss))):.1f}  σ={q_sac_ss.std():.2f}%",
        transform=ax.transAxes, va="top", fontsize=9.5,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7))

# Histograma PI
ax = axes[1, 0]
ax.hist(q_pi_ss, bins=30, color=C_PI, alpha=0.75, edgecolor="white")
ax.set_xlabel("$Q_u$ (%)")
ax.set_ylabel("Frecuencia")
ax.set_title("Histograma $Q_u$ — PI")
ax.axvline(q_pi_ss.mean(), color="black", lw=1.5, ls="--", label=f"Media={q_pi_ss.mean():.1f}%")
ax.legend(fontsize=9)

# Histograma SAC
ax = axes[1, 1]
ax.hist(q_sac_ss, bins=30, color=C_SAC, alpha=0.75, edgecolor="white")
ax.set_xlabel("$Q_u$ (%)")
ax.set_ylabel("Frecuencia")
ax.set_title("Histograma $Q_u$ — SAC")
ax.axvline(q_sac_ss.mean(), color="black", lw=1.5, ls="--", label=f"Media={q_sac_ss.mean():.1f}%")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig5_chattering_analysis.pdf")
plt.savefig(f"{OUT_DIR}/fig5_chattering_analysis.png")
plt.close()
print("[OK] fig5_chattering_analysis guardada")


# FIGURA 6: Energía Acumulada y ISE Acumulado
cum_energy_pi  = np.cumsum(Q_pi)
cum_energy_sac = np.cumsum(Q_sac)
cum_ise_pi  = np.cumsum(e_pi**2)
cum_ise_sac = np.cumsum(e_sac**2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Integrales Acumuladas — Energía Total e ISE", fontweight="bold")

add_phase_bands(ax1)
ax1.plot(t_pi,  cum_energy_pi,  color=C_PI,  lw=LW, alpha=ALPHA,
         label=f"PI: {cum_energy_pi[-1]:.0f} %·s")
ax1.plot(t_sac, cum_energy_sac, color=C_SAC, lw=LW, alpha=ALPHA,
         label=f"SAC: {cum_energy_sac[-1]:.0f} %·s")
ax1.set_xlabel("Tiempo (s)")
ax1.set_ylabel("$\\int Q_u \\, dt$ (%·s)")
ax1.set_title("Energía Total Acumulada")
ax1.legend(framealpha=0.9)

add_phase_bands(ax2)
ax2.plot(t_pi,  cum_ise_pi,  color=C_PI,  lw=LW, alpha=ALPHA,
         label=f"PI: ISE={cum_ise_pi[-1]:.0f}")
ax2.plot(t_sac, cum_ise_sac, color=C_SAC, lw=LW, alpha=ALPHA,
         label=f"SAC: ISE={cum_ise_sac[-1]:.0f}")
ax2.set_xlabel("Tiempo (s)")
ax2.set_ylabel("$\\int e^2(t) \\, dt$")
ax2.set_title("ISE Acumulado (Error Integral Cuadrático)")
ax2.legend(framealpha=0.9)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig6_integrals.pdf")
plt.savefig(f"{OUT_DIR}/fig6_integrals.png")
plt.close()
print("[OK] fig6_integrals guardada")


# FIGURA 7: Diagrama del Espacio de Observación del Agente SAC
fig, ax = plt.subplots(figsize=(13, 4))
ax.axis("off")
ax.set_title("Espacio de Observación del Agente SAC — 46 Dimensiones", fontweight="bold", y=1.05)

groups = [
    ("Estados\nde Error\n(5 dim)", 5, "#AED6F1"),
    ("Temp.\nAmbiente\n(1 dim)", 1, "#A9DFBF"),
    ("Historial\nAcciones\n(25 dim)", 25, "#FAD7A0"),
    ("Preview\nReferencia\n(15 dim)", 15, "#D7BDE2"),
]

total = sum(g[1] for g in groups)
x = 0.0
for label, ndim, color in groups:
    width = ndim / total
    rect = plt.Rectangle((x, 0.15), width-0.005, 0.7, color=color,
                           transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)
    ax.text(x + width/2, 0.5, f"{label}\n[{ndim}]",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, fontweight="bold")
    x += width

# Anotaciones bajo el diagrama
ax.text(0.5/46, -0.12, "$e_T,\\;\\int e_T,\\;\\dot{e}_T,\\;T_{obs},\\;T^*$",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#1A5276")
ax.text((5+0.5)/46, -0.12, "$T_a$", transform=ax.transAxes,
        ha="center", va="top", fontsize=8.5, color="#1E8449")
ax.text((6+12.5)/46, -0.12, "$Q_{u,t-1},\\ldots,Q_{u,t-25}$",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#784212")
ax.text((31+7.5)/46, -0.12, "$T^*_{t+1},\\ldots,T^*_{t+15}$",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#4A235A")

plt.savefig(f"{OUT_DIR}/fig7_obs_space.pdf", bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig7_obs_space.png", bbox_inches="tight")
plt.close()
print("[OK] fig7_obs_space guardada")


# FIGURA 8: Diagrama de la Arquitectura SAC (Actor-Crítico)
fig, ax = plt.subplots(figsize=(11, 5))
ax.axis("off")
ax.set_title("Arquitectura del Agente SAC para Control Térmico", fontweight="bold")

def draw_box(ax, x, y, w, h, label, color="#AED6F1", fontsize=9):
    rect = plt.Rectangle((x-w/2, y-h/2), w, h, fc=color, ec="gray", lw=1.3, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", zorder=4, wrap=True)

def draw_arrow(ax, x1, x2, y, label="", color="black"):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5), zorder=5)
    if label:
        ax.text((x1+x2)/2, y+0.04, label, ha="center", fontsize=8, color=color)

ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# Observación
draw_box(ax, 0.08, 0.70, 0.12, 0.14, "Obs. $s_t$\n(46 dim)", "#D5DBDB")
draw_box(ax, 0.08, 0.30, 0.12, 0.14, "Obs. $s_t$\n(46 dim)", "#D5DBDB")

# Actor
draw_box(ax, 0.35, 0.70, 0.22, 0.16,
         "Actor $\\pi_\\theta$\n[64]→[64]\nReLU, LayerNorm", "#AED6F1")
# Críticos
draw_box(ax, 0.35, 0.38, 0.22, 0.12, "Critic $Q_{\\phi_1}$\n[64]→[64]", "#FAD7A0")
draw_box(ax, 0.35, 0.22, 0.22, 0.12, "Critic $Q_{\\phi_2}$\n[64]→[64]", "#FAD7A0")

# Salida Actor
draw_box(ax, 0.64, 0.70, 0.14, 0.14,
         "$(\\mu, \\log\\sigma)$\n→ $a_t \\sim \\pi$", "#A9DFBF")

# Acción → entorno
draw_box(ax, 0.87, 0.70, 0.12, 0.14,
         "Planta\n$Q_u = 50(a_t+1)$", "#D7BDE2")

# Temperatura de entropía
draw_box(ax, 0.64, 0.38, 0.14, 0.12,
         "$\\alpha_e$ auto\n$\\mathcal{H}_{target}=-1$", "#FADBD8")

# Flechas
draw_arrow(ax, 0.14, 0.24, 0.70, "$s_t$")
draw_arrow(ax, 0.14, 0.24, 0.30, "$s_t$, $a_t$")
draw_arrow(ax, 0.46, 0.57, 0.70, "$\\mu,\\sigma$")
draw_arrow(ax, 0.71, 0.81, 0.70, "$a_t$")
draw_arrow(ax, 0.46, 0.57, 0.38)
draw_arrow(ax, 0.46, 0.57, 0.22)

# Labels
ax.text(0.35, 0.88, "ACTOR (política estocástica)", ha="center", fontsize=9.5,
        color="#1A5276", style="italic")
ax.text(0.35, 0.12, "CRÍTICOS (evaluación de valor)", ha="center", fontsize=9.5,
        color="#784212", style="italic")

plt.savefig(f"{OUT_DIR}/fig8_sac_architecture.pdf", bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig8_sac_architecture.png", bbox_inches="tight")
plt.close()
print("[OK] fig8_sac_architecture guardada")


# FIGURA 9: Diagrama del Curriculum Learning
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.axis("off")
ax.set_title("Protocolo de Curriculum Learning — Progresión de Complejidad Paramétrica",
             fontweight="bold")

stages = [
    ("Etapa 1\n(300s)", "$T^*=40$°C fijo\n$\\theta=0$ s\nDR=0%\n$P_{cold}=0\%$", "#EBF5FB"),
    ("Etapa 2\n(500s)", "$T^*\\in[35,45]$°C\n$\\theta=11$ s\nDR=5%\n$P_{cold}=20\%$", "#D4EFDF"),
    ("Etapa 3\n(500s)", "$T^*\\in[30,48]$°C\n$\\theta\\in[0,22]$ s\nDR=10%\n$P_{cold}=40\%$", "#FDEBD0"),
    ("Etapa 4\n(500s)", "$T^*\\in[30,48]$°C\n$\\theta\\in[0,33]$ s\nDR=15%\n$P_{cold}=50\%$", "#FDEDEC"),
]
thresholds = ["-166", "-200", "-340"]

xpos = [0.12, 0.35, 0.62, 0.87]
for i, (title, desc, color) in enumerate(stages):
    rect = plt.Rectangle((xpos[i]-0.115, 0.12), 0.23, 0.78, fc=color, ec="gray", lw=1.5,
                           transform=ax.transAxes, clip_on=False, zorder=2)
    ax.add_patch(rect)
    ax.text(xpos[i], 0.87, title, transform=ax.transAxes,
            ha="center", va="top", fontsize=10.5, fontweight="bold", zorder=3)
    ax.text(xpos[i], 0.65, desc, transform=ax.transAxes,
            ha="center", va="top", fontsize=9, zorder=3, linespacing=1.5)
    if i < 3:
        ax.annotate("", xy=(xpos[i+1]-0.115, 0.5), xytext=(xpos[i]+0.115, 0.5),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2.2), zorder=5)
        ax.text((xpos[i]+xpos[i+1])/2, 0.27, f"$\\bar{{r}}\\geq{thresholds[i]}$",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="darkgreen", style="italic",
                bbox=dict(boxstyle="round", fc="white", ec="darkgreen", alpha=0.85))

plt.savefig(f"{OUT_DIR}/fig9_curriculum.pdf", bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig9_curriculum.png", bbox_inches="tight")
plt.close()
print("[OK] fig9_curriculum guardada")

print(f"\n{'='*55}")
print(f"  Todas las figuras guardadas en: {OUT_DIR}/")
print(f"{'='*55}")
