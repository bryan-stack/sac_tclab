"""
nmpc_sensitivity.py — Análisis de sensibilidad del NMPC (horizonte y pesos)
==========================================================================
Diagnostica en SIMULACIÓN por qué el NMPC con los parámetros heredados del
paper de referencia (S=0.0113, R=0.001, P_N=0.1513, Tp=30) rinde mal, y
busca una sintonía que elimine el sobreimpulso y baje el ISE para el perfil
de estrés de este trabajo.

Barre:
  - Horizonte de predicción Tp ∈ {30, 60, 90, 120}  (Tc=5)
  - Conjuntos de pesos (S, R, P_N)

Evalúa en dos escenarios:
  - 'const' : setpoint constante 40.8 °C, arranque en frío (500 s)
  - 'stress': perfil de 4 fases 35->48->28->40 °C (600 s)

Backend: simulación ODE (PlantSim), N réplicas (semillas distintas), media.
Salida: imprime tabla y guarda nmpc_sensitivity.csv
"""

import numpy as np
import pandas as pd

from stress_suite_v2 import (
    PlantSim, PIController, referencia_dinamica, _load_nmpc_solver, NOMINAL_PARAMS,
)

NMPCSolver = _load_nmpc_solver()


def ref_const(t, sp=40.8):
    return sp


def run_nmpc(profile_fn, duration, Tp, Tc, S, R, P_N, seed, dt=1.0):
    """Corre el NMPC puro (sin caché, recalcula cada paso) sobre PlantSim."""
    solver = NMPCSolver(dt=dt, Tp=Tp, Tc=Tc, S=S, R=R, P_N=P_N)
    plant = PlantSim(T0_C=32.0, Ta_C=32.0, dead_steps=22, noise_std=0.10, dt=dt, seed=seed)
    # sincroniza Ta del modelo interno con la planta
    p = dict(NOMINAL_PARAMS); p["Ta"] = 273.15 + 32.0
    solver.update_params(**p)
    Q_prev = 30.0
    T_log, ref_log, Qu_log = [], [], []
    for k in range(int(duration) + 1):
        t = k * dt
        T_real = plant.T1
        T_ref = profile_fn(t)
        Q = solver.compute_action(T0_C=T_real, T_ref_C=T_ref, Q_warm=Q_prev)
        Q_prev = Q
        Qu = float(np.clip(Q, 0, 100))
        plant.step(Qu)
        T_log.append(T_real); ref_log.append(T_ref); Qu_log.append(Qu)
    return np.array(T_log), np.array(ref_log), np.array(Qu_log)


def run_pi(profile_fn, duration, seed, dt=1.0):
    ctrl = PIController()
    plant = PlantSim(T0_C=32.0, Ta_C=32.0, dead_steps=22, noise_std=0.10, dt=dt, seed=seed)
    ctrl.reset(plant.T1)
    T_log, ref_log, Qu_log = [], [], []
    for k in range(int(duration) + 1):
        t = k * dt
        T_real = plant.T1
        T_ref = profile_fn(t)
        Qu = ctrl.action(T_real, T_ref)
        plant.step(Qu)
        T_log.append(T_real); ref_log.append(T_ref); Qu_log.append(Qu)
    return np.array(T_log), np.array(ref_log), np.array(Qu_log)


def metrics(T, ref, Qu, dt=1.0):
    e = ref - T
    ise = float(np.sum(e ** 2))
    iae = float(np.sum(np.abs(e)))
    t = np.arange(len(e)) * dt
    itae = float(np.sum(t * np.abs(e)))
    os_max = float(max(0.0, (T - ref).max()))
    energy = float(np.sum(Qu))
    tv = float(np.sum(np.abs(np.diff(Qu))))
    return dict(ISE=ise, IAE=iae, ITAE=itae, OS=os_max, Energy=energy, TV=tv)


def avg_metrics(runner_kwargs, profile_fn, duration, reps=3):
    accs = []
    for i in range(reps):
        T, ref, Qu = runner_kwargs(profile_fn, duration, seed=1000 + i)
        accs.append(metrics(T, ref, Qu))
    keys = accs[0].keys()
    return {k: float(np.mean([a[k] for a in accs])) for k in keys}


def main():
    reps = 3
    rows = []

    # ---- Baselines PI ----
    for scen, prof, dur in [("const", ref_const, 500), ("stress", referencia_dinamica, 600)]:
        m = avg_metrics(lambda pf, d, seed: run_pi(pf, d, seed), prof, dur, reps)
        rows.append(dict(scenario=scen, ctrl="PI", Tp="-", S="-", R="-", P_N="-", **m))

    # ---- Conjuntos de pesos a probar ----
    # baseline = parámetros heredados del paper de referencia
    weight_sets = {
        "ref[7]":     (0.0113, 0.001,  0.1513),
        "S_alta":     (0.05,   0.001,  0.1513),   # penaliza más el error de etapa
        "R_baja":     (0.0113, 0.0001, 0.1513),   # menos penalización al esfuerzo
        "PN_alta":    (0.0113, 0.001,  0.5),      # más peso terminal
        "balance":    (0.05,   0.0005, 0.5),      # combinación
    }
    horizons = [30, 60, 90, 120]

    for scen, prof, dur in [("const", ref_const, 500), ("stress", referencia_dinamica, 600)]:
        for wname, (S, R, P_N) in weight_sets.items():
            for Tp in horizons:
                Tc = 5
                def runner(pf, d, seed, Tp=Tp, Tc=Tc, S=S, R=R, P_N=P_N):
                    return run_nmpc(pf, d, Tp, Tc, S, R, P_N, seed)
                m = avg_metrics(runner, prof, dur, reps)
                rows.append(dict(scenario=scen, ctrl=f"NMPC/{wname}",
                                 Tp=Tp, S=S, R=R, P_N=P_N, **m))
                print(f"[{scen:6s}] {wname:8s} Tp={Tp:3d}  "
                      f"ISE={m['ISE']:8.1f}  OS={m['OS']:5.2f}  "
                      f"IAE={m['IAE']:7.1f}  TV={m['TV']:6.1f}")

    df = pd.DataFrame(rows)
    df.to_csv("nmpc_sensitivity.csv", index=False)
    print("\n[OK] nmpc_sensitivity.csv guardado.")

    # Resumen: mejor combinación por escenario según ISE
    for scen in ["const", "stress"]:
        sub = df[(df.scenario == scen) & (df.ctrl.str.startswith("NMPC"))]
        best = sub.loc[sub.ISE.idxmin()]
        pi = df[(df.scenario == scen) & (df.ctrl == "PI")].iloc[0]
        print(f"\n=== {scen} ===")
        print(f"  PI:          ISE={pi.ISE:8.1f}  OS={pi.OS:5.2f}")
        print(f"  NMPC ref[7]: ", end="")
        ref_row = sub[(sub.ctrl == "NMPC/ref[7]") & (sub.Tp == 30)].iloc[0]
        print(f"ISE={ref_row.ISE:8.1f}  OS={ref_row.OS:5.2f}  (Tp=30)")
        print(f"  NMPC mejor:  ISE={best.ISE:8.1f}  OS={best.OS:5.2f}  "
              f"({best.ctrl}, Tp={best.Tp}, S={best.S}, R={best.R}, P_N={best.P_N})")


if __name__ == "__main__":
    main()
