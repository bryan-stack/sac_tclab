"""
analisis_completo.py — Números exactos para el manuscrito (datos de hardware 4x4)
=================================================================================
Calcula, a partir de las 4 réplicas de hardware de cada controlador:
  - media ± sigma de todas las métricas
  - mejoras relativas del SAC vs cada controlador
  - réplica representativa (mediana de ISE) para las figuras
  - estadísticas por fase de la réplica representativa
  - análisis de windup (tiempo de saturación del PI) y chattering
"""
import sys, glob
import numpy as np
import pandas as pd
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODES = ["pi", "nmpc", "nmpcsac", "sac"]
LBL = {"pi": "PI", "nmpc": "NMPC", "nmpcsac": "NMPC-SAC", "sac": "SAC"}
PHASES = [(0, 150, 35), (150, 300, 48), (300, 450, 28), (450, 600, 40)]


def met(df, dt=1.0):
    t = df["Time(s)"].values
    e = df["Reference(C)"].values - df["Temperature(C)"].values
    u = df["Control_Effort(%)"].values
    return dict(
        ISE=np.sum(e**2)*dt, IAE=np.sum(np.abs(e))*dt, ITAE=np.sum(t*np.abs(e))*dt,
        OS=max(0.0, (df["Temperature(C)"].values-df["Reference(C)"].values).max()),
        Energia=np.sum(u)*dt, TV=np.sum(np.abs(np.diff(u))),
    )

# Cargar todas las réplicas
data = {m: [pd.read_csv(f) for f in sorted(glob.glob(f"stress_{m}_rep*.csv"))] for m in MODES}
metrics = {m: [met(df) for df in data[m]] for m in MODES}
keys = ["ISE", "IAE", "ITAE", "OS", "Energia", "TV"]

# 1) Media y sigma
print("="*70); print("1) MEDIA ± SIGMA (hardware, N=4)"); print("="*70)
mean = {m: {} for m in MODES}; std = {m: {} for m in MODES}
for m in MODES:
    for k in keys:
        arr = np.array([mm[k] for mm in metrics[m]])
        mean[m][k] = arr.mean(); std[m][k] = arr.std(ddof=1)
    print(f"{LBL[m]:<10} " + "  ".join(f"{k}={mean[m][k]:.1f}±{std[m][k]:.1f}" for k in keys))

# 2) Mejoras relativas SAC vs cada uno
print("\n"+"="*70); print("2) MEJORA RELATIVA DEL SAC (- = SAC mejor en error/energía)"); print("="*70)
for ref in ["pi", "nmpc", "nmpcsac"]:
    print(f"\nSAC vs {LBL[ref]}:")
    for k in keys:
        rel = (mean["sac"][k]-mean[ref][k])/mean[ref][k]*100
        print(f"   {k:8}: {rel:+.1f}%   (SAC={mean['sac'][k]:.1f}, {LBL[ref]}={mean[ref][k]:.1f})")

# 3) Réplica representativa (mediana de ISE)
print("\n"+"="*70); print("3) RÉPLICA REPRESENTATIVA (mediana de ISE)"); print("="*70)
rep_idx = {}
for m in MODES:
    ises = np.array([mm["ISE"] for mm in metrics[m]])
    # índice de la réplica más cercana a la mediana
    order = np.argsort(ises)
    rep = order[len(order)//2]
    rep_idx[m] = rep
    print(f"{LBL[m]:<10} rep{rep+1}  (ISE={ises[rep]:.1f}, mediana={np.median(ises):.1f})")

# 4) Estadísticas por fase de la réplica representativa
print("\n"+"="*70); print("4) FASE A FASE (réplica representativa)"); print("="*70)
print(f"{'Fase':<14}{'Ctrl':<10}{'MaxT':>8}{'MinT':>8}{'OS':>8}{'MaxQu':>8}{'MinQu':>8}")
phase_stats = {}
for pi_, (t0, t1, ref) in enumerate(PHASES, 1):
    for m in MODES:
        df = data[m][rep_idx[m]]
        seg = df[(df["Time(s)"]>=t0)&(df["Time(s)"]<t1)]
        T = seg["Temperature(C)"].values; Q = seg["Control_Effort(%)"].values
        os_ = max(0.0, (T-ref).max())
        phase_stats[(pi_, m)] = dict(maxT=T.max(), minT=T.min(), os=os_, maxq=Q.max(), minq=Q.min())
        print(f"{f'{pi_} ({ref}°C)':<14}{LBL[m]:<10}{T.max():>8.2f}{T.min():>8.2f}{os_:>8.2f}{Q.max():>8.1f}{Q.min():>8.1f}")
    print()

# 5) Windup: tiempo de saturación del PI en fase 2 (rep representativa)
print("="*70); print("5) ANÁLISIS DE WINDUP Y CHATTERING (rep representativa)"); print("="*70)
df_pi = data["pi"][rep_idx["pi"]]
seg2 = df_pi[(df_pi["Time(s)"]>=150)&(df_pi["Time(s)"]<300)]
t_sat = int((seg2["Control_Effort(%)"].values >= 99.0).sum())
print(f"PI: pasos con Qu>=99% en fase 2: {t_sat}s")
# Chattering en cuasi-estacionario t in [50,140]
for m in ["pi", "sac"]:
    df = data[m][rep_idx[m]]
    seg = df[(df["Time(s)"]>=50)&(df["Time(s)"]<140)]
    q = seg["Control_Effort(%)"].values
    print(f"{LBL[m]}: std(Qu) en t∈[50,140]={q.std():.2f}%, TV_seg={np.sum(np.abs(np.diff(q))):.1f}")

# 6) Exportar réplica representativa a CSV "_rep" para las figuras
print("\n"+"="*70); print("6) EXPORTANDO RÉPLICAS REPRESENTATIVAS PARA FIGURAS"); print("="*70)
for m in MODES:
    data[m][rep_idx[m]].to_csv(f"repr_{m}.csv", index=False)
    print(f"  repr_{m}.csv  <- stress_{m}_rep{rep_idx[m]+1}.csv")
