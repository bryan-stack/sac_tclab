"""
generar_figuras_4ctrl.py — Figuras de los 4 controladores (hardware, N=4)
=========================================================================
Regenera las figuras de la prueba de estrés dinámico incluyendo los CUATRO
controladores (PI, NMPC, NMPC-SAC, SAC) y añade la figura estadística de
barras con media ± desviación estándar sobre las 4 réplicas.

Usa repr_<modo>.csv (réplica representativa = mediana de ISE) para las
trayectorias y stress_<modo>_rep*.csv para las barras estadísticas.

Ejecutar DESPUÉS de analisis_completo.py (que genera los repr_*.csv).
Salida: ./figuras_articulo/
"""
import os, glob, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = "./figuras_articulo"
os.makedirs(OUT, exist_ok=True)

# Colores y etiquetas
C = {"pi": "#d62728", "nmpc": "#ff7f0e", "nmpcsac": "#9467bd", "sac": "#1f77b4"}
LBL = {"pi": "PI Clásico", "nmpc": "NMPC puro", "nmpcsac": "NMPC-SAC", "sac": "SAC Puro"}
ORDER = ["pi", "nmpc", "nmpcsac", "sac"]
PHASES = [(0, 150, 35), (150, 300, 48), (300, 450, 28), (450, 600, 40)]
PHASE_BANDS = ["#d4e6f1", "#fde8d8", "#e8f8e8", "#fef9e7"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "legend.fontsize": 9.5, "figure.dpi": 150,
    "savefig.dpi": 400, "savefig.bbox": "tight", "axes.grid": True,
    "grid.linestyle": ":", "grid.alpha": 0.6,
})

# Cargar trayectorias representativas y réplicas
repr_df = {m: pd.read_csv(f"repr_{m}.csv") for m in ORDER}
reps = {m: [pd.read_csv(f) for f in sorted(glob.glob(f"stress_{m}_rep*.csv"))] for m in ORDER}


def bands(ax):
    for (t0, t1, _), col in zip(PHASES, PHASE_BANDS):
        ax.axvspan(t0, t1, alpha=0.18, color=col, zorder=0)


def met(df):
    t = df["Time(s)"].values
    e = df["Reference(C)"].values - df["Temperature(C)"].values
    u = df["Control_Effort(%)"].values
    return dict(ISE=np.sum(e**2), IAE=np.sum(np.abs(e)), ITAE=np.sum(t*np.abs(e)),
               OS=max(0.0, (df["Temperature(C)"].values-df["Reference(C)"].values).max()),
               Energia=np.sum(u), TV=np.sum(np.abs(np.diff(u))))


# FIG 1: Comparativa principal — 4 controladores
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})
bands(ax1); bands(ax2)
ref = repr_df["pi"]
ax1.plot(ref["Time(s)"], ref["Reference(C)"], "k--", lw=1.8, label="Referencia $T^*(t)$", zorder=6)
for m in ORDER:
    d = repr_df[m]
    ax1.plot(d["Time(s)"], d["Temperature(C)"], color=C[m], lw=2.0, alpha=0.9, label=LBL[m])
for (t0, t1, _), lab in zip(PHASES, ["Fase 1","Fase 2","Fase 3","Fase 4"]):
    ax1.text((t0+t1)/2, 53.2, lab, ha="center", fontsize=9, color="gray", style="italic")
ax1.set_ylabel("Temperatura (°C)"); ax1.set_ylim(27, 54)
ax1.legend(loc="lower center", ncol=5, framealpha=0.9, fontsize=9)
ax1.set_title("Prueba de Estrés Dinámico en Hardware Real — Comparativa de 4 Controladores "
              "(réplica representativa)", fontweight="bold")
for m in ORDER:
    d = repr_df[m]
    ax2.step(d["Time(s)"], d["Control_Effort(%)"], color=C[m], lw=1.2, alpha=0.8,
             where="post", label=LBL[m])
ax2.set_xlabel("Tiempo (s)"); ax2.set_ylabel("Esfuerzo de\nControl (%)"); ax2.set_ylim(-5, 108)
ax2.legend(loc="upper right", ncol=4, framealpha=0.9, fontsize=8.5)
for ax in (ax1, ax2):
    ax.xaxis.set_major_locator(MultipleLocator(50)); ax.xaxis.set_minor_locator(MultipleLocator(10))
plt.savefig(f"{OUT}/fig1_comparativa_principal.pdf"); plt.savefig(f"{OUT}/fig1_comparativa_principal.png")
plt.close(); print("[OK] fig1_comparativa_principal (4 ctrl)")


# FIG 2: Error de seguimiento — 4 controladores
fig, ax = plt.subplots(figsize=(12, 4.3))
bands(ax)
ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.5)
for m in ORDER:
    d = repr_df[m]
    e = d["Reference(C)"].values - d["Temperature(C)"].values
    ax.plot(d["Time(s)"], e, color=C[m], lw=1.8, alpha=0.85, label=LBL[m])
ax.set_xlabel("Tiempo (s)"); ax.set_ylabel("Error $e(t)=T^*-T$ (°C)")
ax.set_title("Trayectoria del Error de Seguimiento — 4 Controladores", fontweight="bold")
ax.legend(loc="lower right", ncol=4, framealpha=0.9)
ax.xaxis.set_major_locator(MultipleLocator(50))
plt.savefig(f"{OUT}/fig2_error_trajectory.pdf"); plt.savefig(f"{OUT}/fig2_error_trajectory.png")
plt.close(); print("[OK] fig2_error_trajectory (4 ctrl)")


# FIG 3: Zoom windup (fase 2) — 4 controladores
Z0, Z1 = 130, 320
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1], "hspace": 0.1})
ax1.axvspan(150, 300, alpha=0.08, color="#fde8d8")
ax1.axvline(150, color="gray", lw=1.2, ls="--")
d = repr_df["pi"]; m_ = (d["Time(s)"]>=Z0)&(d["Time(s)"]<=Z1)
ax1.plot(d["Time(s)"][m_], d["Reference(C)"][m_], "k--", lw=1.8)
for m in ORDER:
    d = repr_df[m]; mm = (d["Time(s)"]>=Z0)&(d["Time(s)"]<=Z1)
    ax1.plot(d["Time(s)"][mm], d["Temperature(C)"][mm], color=C[m], lw=2.0, alpha=0.9, label=LBL[m])
ax1.set_ylabel("Temperatura (°C)")
ax1.set_title("Zoom: Escalón de Calentamiento +13°C ($t=150$s) — Windup y Anticipación",
              fontweight="bold")
ax1.legend(loc="lower right", ncol=2, framealpha=0.9)
ax2.axvline(150, color="gray", lw=1.2, ls="--")
ax2.axhline(100, color="gray", lw=0.8, ls=":", alpha=0.6)
for m in ORDER:
    d = repr_df[m]; mm = (d["Time(s)"]>=Z0)&(d["Time(s)"]<=Z1)
    ax2.step(d["Time(s)"][mm], d["Control_Effort(%)"][mm], color=C[m], lw=1.3,
             where="post", alpha=0.85)
ax2.set_xlabel("Tiempo (s)"); ax2.set_ylabel("$Q_u$ (%)"); ax2.set_ylim(-5, 110)
plt.savefig(f"{OUT}/fig3_zoom_windup.pdf"); plt.savefig(f"{OUT}/fig3_zoom_windup.png")
plt.close(); print("[OK] fig3_zoom_windup (4 ctrl)")


# FIG 4: Zoom enfriamiento pasivo (fase 3) — 4 controladores
Z0, Z1 = 290, 460
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1], "hspace": 0.1})
ax1.axvspan(300, 450, alpha=0.08, color="#e8f8e8")
ax1.axvline(300, color="gray", lw=1.2, ls="--")
ax1.axhline(28, color="green", lw=1.0, ls=":", alpha=0.7, label="Ref = 28°C")
ax1.axhline(32, color="brown", lw=1.0, ls="-.", alpha=0.6, label="$T_a\\approx32$°C (inalcanzable)")
for m in ORDER:
    d = repr_df[m]; mm = (d["Time(s)"]>=Z0)&(d["Time(s)"]<=Z1)
    ax1.plot(d["Time(s)"][mm], d["Temperature(C)"][mm], color=C[m], lw=2.0, alpha=0.9, label=LBL[m])
ax1.set_ylabel("Temperatura (°C)")
ax1.set_title("Zoom: Enfriamiento Pasivo ($T^*=28$°C $<T_a$) — Asimetría Térmica",
              fontweight="bold")
ax1.legend(loc="upper right", ncol=2, framealpha=0.9, fontsize=8.5)
ax2.axvline(300, color="gray", lw=1.2, ls="--")
for m in ORDER:
    d = repr_df[m]; mm = (d["Time(s)"]>=Z0)&(d["Time(s)"]<=Z1)
    ax2.step(d["Time(s)"][mm], d["Control_Effort(%)"][mm], color=C[m], lw=1.3,
             where="post", alpha=0.85)
ax2.set_xlabel("Tiempo (s)"); ax2.set_ylabel("$Q_u$ (%)"); ax2.set_ylim(-3, 45)
plt.savefig(f"{OUT}/fig4_zoom_cooling.pdf"); plt.savefig(f"{OUT}/fig4_zoom_cooling.png")
plt.close(); print("[OK] fig4_zoom_cooling (4 ctrl)")


# FIG 6: Integrales acumuladas — 4 controladores
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Integrales Acumuladas — Energía e ISE (réplica representativa)", fontweight="bold")
bands(ax1); bands(ax2)
for m in ORDER:
    d = repr_df[m]
    ax1.plot(d["Time(s)"], np.cumsum(d["Control_Effort(%)"].values), color=C[m], lw=2.0,
             alpha=0.88, label=f"{LBL[m]}: {np.sum(d['Control_Effort(%)'].values):.0f}")
    e = d["Reference(C)"].values - d["Temperature(C)"].values
    ax2.plot(d["Time(s)"], np.cumsum(e**2), color=C[m], lw=2.0, alpha=0.88,
             label=f"{LBL[m]}: {np.sum(e**2):.0f}")
ax1.set_xlabel("Tiempo (s)"); ax1.set_ylabel("$\\int Q_u\\,dt$ (%·s)"); ax1.set_title("Energía Total Acumulada")
ax1.legend(framealpha=0.9, fontsize=8.5)
ax2.set_xlabel("Tiempo (s)"); ax2.set_ylabel("$\\int e^2\\,dt$"); ax2.set_title("ISE Acumulado")
ax2.legend(framealpha=0.9, fontsize=8.5)
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_integrals.pdf"); plt.savefig(f"{OUT}/fig6_integrals.png")
plt.close(); print("[OK] fig6_integrals (4 ctrl)")


# FIG 10 (NUEVA): Barras estadísticas media ± sigma sobre las 4 réplicas
metric_keys = ["ISE", "IAE", "ITAE", "OS", "Energia", "TV"]
metric_titles = {"ISE": "ISE", "IAE": "IAE", "ITAE": "ITAE",
                 "OS": "Sobreimpulso máx. (°C)", "Energia": "Energía (%·s)", "TV": "Variación Total"}
agg = {m: {k: [] for k in metric_keys} for m in ORDER}
for m in ORDER:
    for df in reps[m]:
        mt = met(df)
        for k in metric_keys:
            agg[m][k].append(mt[k])

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
fig.suptitle("Métricas de Desempeño en Hardware — Media ± Desviación Estándar (N=4 réplicas)",
             fontweight="bold", fontsize=13)
for ax, k in zip(axes.flat, metric_keys):
    means = [np.mean(agg[m][k]) for m in ORDER]
    stds = [np.std(agg[m][k], ddof=1) for m in ORDER]
    colors = [C[m] for m in ORDER]
    bars = ax.bar(range(4), means, yerr=stds, capsize=5, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.7)
    ax.set_xticks(range(4)); ax.set_xticklabels([LBL[m].replace(" ", "\n") for m in ORDER], fontsize=8.5)
    ax.set_title(metric_titles[k], fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    # resaltar el SAC (mejor en error)
    for i, m in enumerate(ORDER):
        if m == "sac":
            bars[i].set_edgecolor("navy"); bars[i].set_linewidth(2.0)
plt.tight_layout()
plt.savefig(f"{OUT}/fig10_barras_estadisticas.pdf"); plt.savefig(f"{OUT}/fig10_barras_estadisticas.png")
plt.close(); print("[OK] fig10_barras_estadisticas (NUEVA)")

# FIG 5: Chattering PI vs SAC en cuasi-estacionario (datos nuevos)
Z0, Z1 = 50, 145
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
fig.suptitle("Análisis de Chattering de la Señal de Control "
             "(Estado Cuasi-Estacionario, $t\\in[50,145]$s)", fontweight="bold")
for col, m in enumerate(["pi", "sac"]):
    d = repr_df[m]; seg = d[(d["Time(s)"]>=Z0)&(d["Time(s)"]<=Z1)]
    q = seg["Control_Effort(%)"].values
    ax = axes[0, col]
    ax.step(seg["Time(s)"], q, color=C[m], lw=1.2, where="post")
    ax.set_title(f"Señal $Q_u(t)$ — {LBL[m]}"); ax.set_ylabel("$Q_u$ (%)"); ax.set_xlabel("Tiempo (s)")
    ax.text(0.02, 0.95, f"TV={np.sum(np.abs(np.diff(q))):.1f}  σ={q.std():.2f}%",
            transform=ax.transAxes, va="top", fontsize=9.5,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    ax = axes[1, col]
    ax.hist(q, bins=30, color=C[m], alpha=0.75, edgecolor="white")
    ax.axvline(q.mean(), color="black", lw=1.5, ls="--", label=f"Media={q.mean():.1f}%")
    ax.set_xlabel("$Q_u$ (%)"); ax.set_ylabel("Frecuencia"); ax.set_title(f"Histograma $Q_u$ — {LBL[m]}")
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_chattering_analysis.pdf"); plt.savefig(f"{OUT}/fig5_chattering_analysis.png")
plt.close(); print("[OK] fig5_chattering_analysis (datos nuevos)")

print(f"\nFiguras de 4 controladores guardadas en {OUT}/")
