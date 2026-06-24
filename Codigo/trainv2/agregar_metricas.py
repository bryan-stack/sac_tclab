"""
agregar_metricas.py — Agregación Estadística de Réplicas
========================================================
Procesa las réplicas stress_<modo>_rep<i>.csv de los cuatro controladores
y produce una tabla con MEDIA ± DESVIACIÓN ESTÁNDAR de cada métrica, más
el test de Wilcoxon rank-sum (Mann-Whitney U) del SAC contra cada
controlador para verificar significancia estadística (corrección I1).

USO:
    python agregar_metricas.py --dir sim_runs          # datos de simulación
    python agregar_metricas.py --dir .                 # datos de hardware

Salida:
    - Tabla por pantalla (media ± std de ISE, IAE, ITAE, OS, Energía, TV)
    - tabla_estadistica_<dir>.csv
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")   # consola Windows (cp1252)
except Exception:
    pass

try:
    from scipy.stats import mannwhitneyu
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

MODES = ["pi", "nmpc", "nmpcsac", "sac"]
MODE_LABEL = {
    "pi": "PI Clásico",
    "nmpc": "NMPC puro",
    "nmpcsac": "NMPC-SAC",
    "sac": "SAC Puro",
}


def metricas_de_csv(path, dt=1.0):
    df = pd.read_csv(path)
    t = df["Time(s)"].values
    e = df["Reference(C)"].values - df["Temperature(C)"].values
    u = df["Control_Effort(%)"].values
    return {
        "ISE":  float(np.sum(e**2) * dt),
        "IAE":  float(np.sum(np.abs(e)) * dt),
        "ITAE": float(np.sum(t * np.abs(e)) * dt),
        "OS":   float(max(0.0, (df["Temperature(C)"].values - df["Reference(C)"].values).max())),
        "Energia": float(np.sum(u) * dt),
        "TV":   float(np.sum(np.abs(np.diff(u)))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="sim_runs")
    args = ap.parse_args()

    metric_keys = ["ISE", "IAE", "ITAE", "OS", "Energia", "TV"]
    raw = {m: {k: [] for k in metric_keys} for m in MODES}
    counts = {}

    for m in MODES:
        files = sorted(glob.glob(os.path.join(args.dir, f"stress_{m}_rep*.csv")))
        counts[m] = len(files)
        for f in files:
            mt = metricas_de_csv(f)
            for k in metric_keys:
                raw[m][k].append(mt[k])

    print("\n" + "=" * 90)
    print(f"  MÉTRICAS AGREGADAS (media ± σ)  —  directorio: {args.dir}/")
    print("=" * 90)
    header = f"{'Controlador':<14}" + "".join(f"{k:>13}" for k in metric_keys) + f"{'N':>5}"
    print(header)
    print("-" * 90)

    rows = []
    for m in MODES:
        if counts[m] == 0:
            print(f"{MODE_LABEL[m]:<14}  (sin réplicas en {args.dir}/)")
            continue
        cells = []
        row = {"Controlador": MODE_LABEL[m], "N": counts[m]}
        for k in metric_keys:
            arr = np.array(raw[m][k])
            mean, std = arr.mean(), arr.std(ddof=1) if len(arr) > 1 else 0.0
            row[f"{k}_mean"] = round(mean, 2)
            row[f"{k}_std"] = round(std, 2)
            cells.append(f"{mean:>8.1f}±{std:<4.1f}")
        print(f"{MODE_LABEL[m]:<14}" + "".join(f"{c:>13}" for c in cells) + f"{counts[m]:>5}")
        rows.append(row)

    print("=" * 90)

    # Test de Wilcoxon (Mann-Whitney U) del SAC vs cada controlador, sobre ISE
    if _HAS_SCIPY and counts.get("sac", 0) > 1:
        print("\nTest de Mann-Whitney U (SAC vs. otros, métrica ISE):")
        print("-" * 60)
        sac_ise = np.array(raw["sac"]["ISE"])
        for m in ["pi", "nmpc", "nmpcsac"]:
            if counts.get(m, 0) > 1:
                other = np.array(raw[m]["ISE"])
                try:
                    stat, p = mannwhitneyu(sac_ise, other, alternative="less")
                    sig = "SÍ (p<0.05)" if p < 0.05 else "no"
                    print(f"  SAC < {MODE_LABEL[m]:<12}: U={stat:.1f}, p={p:.4f}  -> significativo: {sig}")
                except ValueError as e:
                    print(f"  SAC vs {MODE_LABEL[m]}: {e}")
    elif not _HAS_SCIPY:
        print("\n[!] scipy no disponible: se omite el test de Wilcoxon.")
    else:
        print("\n[!] Se requieren >=2 réplicas de SAC para el test estadístico.")

    if rows:
        out = f"tabla_estadistica_{os.path.basename(os.path.normpath(args.dir))}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n[OK] Tabla exportada a {out}")


if __name__ == "__main__":
    main()
