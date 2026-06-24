"""
generar_figura_setpoint_4ctrl.py — Figura combinada de setpoint constante
=========================================================================
Combina la respuesta de los CUATRO controladores (PI, NMPC, NMPC-SAC, SAC)
en una sola figura de setpoint constante (T* ~ 40.8 °C, arranque en frío),
reemplazando las dos figuras separadas anteriores.

Fuentes de datos:
    PI        : hardware_pid_comparison.csv      (col Control_Effort)
    NMPC puro : hardware_nmpc_const.csv          (col Control_Effort)   <- generar con setpoint_const.py
    NMPC-SAC  : ../nmpc/hardware_nmpcsac_stage_4.csv (col Control_Total)
    SAC       : hardware_sac_stage_4.csv         (col Control_Effort)

USO:
    python generar_figura_setpoint_4ctrl.py
    python generar_figura_setpoint_4ctrl.py --nmpc-file sim_nmpc_const_TEST.csv   (validación)

Salida: figuras_articulo/fig_sp_const_4ctrl.png/.pdf
"""

import os
import sys
import argparse
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

C = {"pi": "#d62728", "nmpc": "#ff7f0e", "nmpcsac": "#9467bd", "sac": "#1f77b4"}
LBL = {"pi": "PI Clásico", "nmpc": "NMPC puro", "nmpcsac": "NMPC-SAC", "sac": "SAC Puro"}
ORDER = ["pi", "nmpc", "nmpcsac", "sac"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "legend.fontsize": 9.5, "figure.dpi": 150,
    "savefig.dpi": 400, "savefig.bbox": "tight", "axes.grid": True,
    "grid.linestyle": ":", "grid.alpha": 0.6,
})


def cargar(path):
    """Carga un CSV y normaliza la columna de control a 'Qu'."""
    df = pd.read_csv(path)
    qcol = "Control_Effort(%)" if "Control_Effort(%)" in df.columns else "Control_Total(%)"
    return pd.DataFrame({
        "t": df["Time(s)"].values,
        "ref": df["Reference(C)"].values,
        "T": df["Temperature(C)"].values,
        "Qu": df[qcol].values,
    })


def metr(d):
    t = d["t"].values
    e = d["ref"].values - d["T"].values
    u = d["Qu"].values
    return dict(
        ISE=float(np.sum(e**2)), IAE=float(np.sum(np.abs(e))),
        ITAE=float(np.sum(t * np.abs(e))),
        OS=float(max(0.0, (d["T"].values - d["ref"].values).max())),
        Energia=float(np.sum(u)), TV=float(np.sum(np.abs(np.diff(u)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmpc-file", default="hardware_nmpc_const.csv",
                    help="CSV del NMPC puro en setpoint constante")
    args = ap.parse_args()

    paths = {
        "pi": "hardware_pid_comparison.csv",
        "nmpc": args.nmpc_file,
        "nmpcsac": "../nmpc/hardware_nmpcsac_stage_4.csv",
        "sac": "hardware_sac_stage_4.csv",
    }
    data = {}
    for m in ORDER:
        if not os.path.exists(paths[m]):
            print(f"[X] FALTA {LBL[m]}: {paths[m]}")
            if m == "nmpc":
                print("    -> Genera el dato con:  python setpoint_const.py --mode nmpc --Tp 30")
            return
        data[m] = cargar(paths[m])

    ref0 = float(np.median(data["pi"]["ref"].values))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})

    ax1.axhline(ref0, color="black", lw=1.6, ls="--", label=f"Referencia $T^*\\approx{ref0:.1f}$°C")
    ax1.fill_between([0, 500], ref0 - 0.5, ref0 + 0.5, color="gray", alpha=0.10,
                     label="Banda ±0.5°C")
    for m in ORDER:
        d = data[m]
        ax1.plot(d["t"], d["T"], color=C[m], lw=2.0, alpha=0.9, label=LBL[m])
    ax1.set_ylabel("Temperatura (°C)")
    ax1.set_title("Regulación a Setpoint Constante en Hardware — 4 Controladores "
                  "(arranque en frío)", fontweight="bold")
    ax1.legend(loc="lower right", ncol=2, framealpha=0.9, fontsize=9)
    # El NMPC sobrepasa ~6 °C; el límite superior debe cubrir el pico de TODOS
    # los controladores para que ninguna curva se salga de la gráfica.
    y_lo = min(d["T"].min() for d in data.values()) - 1.5
    y_hi = max(d["T"].max() for d in data.values()) + 1.5
    ax1.set_ylim(y_lo, y_hi)

    for m in ORDER:
        d = data[m]
        ax2.step(d["t"], d["Qu"], color=C[m], lw=1.3, alpha=0.85, where="post", label=LBL[m])
    ax2.set_xlabel("Tiempo (s)"); ax2.set_ylabel("$Q_u$ (%)")
    ax2.legend(loc="upper right", ncol=4, framealpha=0.9, fontsize=8.5)
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))

    plt.savefig(f"{OUT}/fig_sp_const_4ctrl.pdf")
    plt.savefig(f"{OUT}/fig_sp_const_4ctrl.png")
    plt.close()
    print(f"[OK] fig_sp_const_4ctrl guardada en {OUT}/")

    # Tabla de métricas para el .tex
    print("\nMétricas de setpoint constante (para la tabla del paper):")
    print(f"{'Controlador':<12}{'ISE':>9}{'IAE':>9}{'ITAE':>10}{'OS':>7}{'Energía':>9}{'TV':>8}")
    for m in ORDER:
        x = metr(data[m])
        print(f"{LBL[m]:<12}{x['ISE']:>9.1f}{x['IAE']:>9.1f}{x['ITAE']:>10.0f}"
              f"{x['OS']:>7.2f}{x['Energia']:>9.0f}{x['TV']:>8.1f}")


if __name__ == "__main__":
    main()
