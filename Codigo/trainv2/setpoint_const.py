"""Prueba de regulación a setpoint constante desde arranque en frío.

Corre un controlador (PI, NMPC o SAC) hacia un setpoint fijo (40.8 °C por
defecto) partiendo de temperatura ambiente, para comparar el transitorio de
arranque. Por defecto usa el hardware (tclab); con --sim usa el modelo ODE.
Guarda un CSV con tiempo, referencia, temperatura y esfuerzo de control.

    python setpoint_const.py --mode nmpc --Tp 30
"""

import argparse
import numpy as np
import pandas as pd

from stress_suite_v2 import PlantSim, PIController, NMPCController, SACController


def construir(mode, model_path, Tp):
    if mode == "pi":
        return PIController()
    if mode == "nmpc":
        return NMPCController(Tp=Tp, Tc=5)
    if mode == "sac":
        return SACController(model_path or "models/sac_tclab_v4_best/best_model")
    raise ValueError(mode)


def run_const(mode, backend, ref, duration, dt, model_path, Tp, outfile, seed):
    ctrl = construir(mode, model_path, Tp)
    log_t, log_ref, log_T, log_Qu = [], [], [], []

    if backend == "sim":
        plant = PlantSim(T0_C=32.0, Ta_C=32.0, dead_steps=22,
                         noise_std=0.10, dt=dt, seed=seed)
        ctrl.reset(plant.T1)
        for k in range(int(duration) + 1):
            t = k * dt
            T_real = plant.T1
            Qu = ctrl.action(T_real, ref)
            plant.step(Qu)
            log_t.append(t); log_ref.append(ref)
            log_T.append(round(T_real, 3)); log_Qu.append(Qu)
    else:
        import tclab
        from tclab import clock
        with tclab.TCLab() as lab:
            T0 = lab.T1
            print(f"  [+] T inicial: {T0:.2f} °C (verifica arranque en frío ~30-33 °C)")
            ctrl.reset(T0)
            for t in clock(duration, dt):
                T_real = lab.T1
                Qu = ctrl.action(T_real, ref)
                lab.Q1(Qu)
                log_t.append(t); log_ref.append(ref)
                log_T.append(round(T_real, 3)); log_Qu.append(Qu)
                if t % 30 == 0:
                    print(f"    t={t:4.0f}s  ref={ref:5.2f}  T={T_real:6.2f}  Qu={Qu:5.1f}%")

    df = pd.DataFrame({
        "Time(s)": log_t, "Reference(C)": log_ref,
        "Temperature(C)": log_T, "Control_Effort(%)": log_Qu,
    })
    df.to_csv(outfile, index=False)

    e = df["Reference(C)"].values - df["Temperature(C)"].values
    ise = float(np.sum(e**2))
    os_max = float(max(0.0, (df["Temperature(C)"].values - df["Reference(C)"].values).max()))
    energy = float(np.sum(df["Control_Effort(%)"].values))
    tv = float(np.sum(np.abs(np.diff(df["Control_Effort(%)"].values))))
    print(f"  [OK] {outfile}: ISE={ise:.1f}  OS={os_max:.2f}°C  "
          f"Energía={energy:.0f}  TV={tv:.1f}")
    return outfile


def main():
    ap = argparse.ArgumentParser(description="Prueba de setpoint constante (arranque en frío)")
    ap.add_argument("--mode", required=True, choices=["pi", "nmpc", "sac"])
    ap.add_argument("--ref", type=float, default=40.8, help="Setpoint constante (°C)")
    ap.add_argument("--duration", type=int, default=500)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--Tp", type=int, default=30, help="Horizonte NMPC (>=theta=22)")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--outfile", type=str, default=None)
    args = ap.parse_args()

    backend = "sim" if args.sim else "hw"
    outfile = args.outfile or f"hardware_{args.mode}_const.csv"
    print(f"\n{'='*58}")
    print(f"  SETPOINT CONSTANTE — modo={args.mode.upper()} | backend={backend}")
    print(f"  Ref={args.ref}°C  dur={args.duration}s" +
          (f"  Tp={args.Tp}s" if args.mode == "nmpc" else ""))
    print(f"{'='*58}")
    run_const(args.mode, backend, args.ref, args.duration, args.dt,
              args.model, args.Tp, outfile, seed=777)


if __name__ == "__main__":
    main()
