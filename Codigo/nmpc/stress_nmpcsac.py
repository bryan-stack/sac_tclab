"""Prueba de estrés de la arquitectura residual NMPC-SAC.

Ejecuta el perfil de estrés de 600 s para Qu = clip(Q_NMPC + Δ_SAC, 0, 100),
con --reps réplicas. Está en nmpc/ porque usa el entorno de observación de 56
dimensiones, distinto del SAC puro. Por defecto corre en hardware (tclab); con
--sim usa el modelo ODE. Guarda stress_nmpcsac_rep<i>.csv con el tiempo, la
referencia, la temperatura, el esfuerzo total y sus componentes NMPC y SAC.

    python stress_nmpcsac.py --reps 4 --model models/nmpc_sac_v2/residual
"""

import argparse
import os
import time
import numpy as np
import pandas as pd

from stable_baselines3 import SAC
from tclab_env import TCLabEnv, nonlinear_thermal_ode, NOMINAL_PARAMS  # nmpc/ (56 dim)


# 1. PERFIL DINÁMICO DE ESTRÉS (idéntico al de stress_suite_v2.py)
def referencia_dinamica(t: float) -> float:
    if t < 150:
        return 35.0
    elif t < 300:
        return 48.0
    elif t < 450:
        return 28.0
    else:
        return 40.0


# 2. PLANTA SIMULADA (solo para --sim)
class PlantSim:
    def __init__(self, T0_C=32.0, Ta_C=32.0, dead_steps=22, noise_std=0.10,
                 dt=1.0, seed=None, dr=0.05):
        self.dt = dt
        self.dead_steps = dead_steps
        self.rng = np.random.default_rng(seed)
        self.params = dict(NOMINAL_PARAMS)
        self.params["Ta"] = 273.15 + Ta_C
        if dr > 0:
            self.params["Uh"] *= self.rng.uniform(1 - dr, 1 + dr)
            self.params["K"]  *= self.rng.uniform(1 - dr, 1 + dr)
        self.noise_std = noise_std
        self.T_K = T0_C + 273.15
        self._buffer = [T0_C] * (dead_steps + 1)

    def step(self, Qu):
        dTdt = nonlinear_thermal_ode(self.T_K, float(np.clip(Qu, 0, 100)), self.params)
        self.T_K += dTdt * self.dt
        T_real = self.T_K - 273.15 + self.rng.normal(0, self.noise_std)
        self._buffer.append(T_real)
        return self._buffer.pop(0) if self.dead_steps > 0 else T_real

    @property
    def T1(self):
        return self._buffer[0]


# 3. ADAPTADOR NMPC-SAC RESIDUAL
class NMPCSACController:
    """Arquitectura residual: Qu = clip(Q_NMPC + Δ_SAC, 0, 100).
    Mantiene el cacheo del NMPC (cada NMPC_REFRESH_STEPS) y el residual ±15 %
    tal como se entrenó el modelo. La observación es de 56 dim."""
    def __init__(self, model_path, dt=1.0):
        self.model = SAC.load(model_path, device="cpu")
        self.env = TCLabEnv(stage=4)   # entorno NMPC-SAC (56 dim, con solver)
        self.dt = dt
        self._last = (30.0, 0.0)       # (Q_NMPC, delta) para logging

    def reset(self, T0_C):
        self.env.reset()
        self.env.T_K = T0_C + 273.15
        self.env.Ta_C = T0_C
        self.env.eT_prev = 0.0
        self.env.eT_integral = 0.0
        self.env._action_history = [0.0] * self.env.HISTORY_LEN
        self.env.Q_NMPC_prev = 30.0
        self.env.Q_NMPC_cached = 30.0
        self.env.step_count = 0
        # Sincroniza el modelo interno del NMPC con la temperatura inicial
        p = dict(NOMINAL_PARAMS)
        p["Ta"] = 273.15 + T0_C
        self.env.nmpc.update_params(**p)
        self._obs = self.env._build_obs(T0_C, 0.0, 0.0)

    def action(self, T_real, T_ref):
        act, _ = self.model.predict(self._obs, deterministic=True)
        self.env.T_ref_C = T_ref

        # NMPC con cacheo (cada NMPC_REFRESH_STEPS pasos)
        if self.env.step_count % self.env.NMPC_REFRESH_STEPS == 0:
            Q_NMPC = self.env.nmpc.compute_action(
                T0_C=T_real, T_ref_C=T_ref, Q_warm=self.env.Q_NMPC_prev)
            self.env.Q_NMPC_cached = Q_NMPC
            self.env.Q_NMPC_prev = Q_NMPC
        else:
            Q_NMPC = self.env.Q_NMPC_cached

        delta = float(act[0]) * self.env.MAX_RESIDUAL
        Qu = float(np.clip(Q_NMPC + delta, 0.0, 100.0))
        self._last = (Q_NMPC, delta)

        # Actualiza historial y errores
        self.env._action_history.append(float(act[0]))
        self.env._action_history.pop(0)
        eT_real = T_ref - T_real
        eT_der = (eT_real - self.env.eT_prev) / self.dt
        self.env.eT_integral += eT_real * self.dt
        self.env.eT_prev = eT_real
        self.env.step_count += 1
        self._obs = self.env._build_obs(T_real, eT_real, eT_der)
        return Qu


# 4. MOTOR DE EJECUCIÓN
def ejecutar_una_corrida(controller, backend, duration, dt, rep_idx, seed, outdir="."):
    log_t, log_ref, log_T, log_Qu, log_Qnmpc, log_delta = [], [], [], [], [], []

    if backend == "sim":
        plant = PlantSim(T0_C=32.0, Ta_C=32.0, dead_steps=22,
                         noise_std=0.10, dt=dt, seed=seed)
        T0 = plant.T1
        controller.reset(T0)
        for k in range(int(duration) + 1):
            t = k * dt
            T_real = plant.T1
            T_ref = referencia_dinamica(t)
            Qu = controller.action(T_real, T_ref)
            Q_nmpc, delta = controller._last
            plant.step(Qu)
            log_t.append(t); log_ref.append(T_ref); log_T.append(round(T_real, 3))
            log_Qu.append(Qu); log_Qnmpc.append(Q_nmpc); log_delta.append(delta)
    else:
        import tclab
        from tclab import clock
        with tclab.TCLab() as lab:
            T0 = lab.T1
            print(f"  [+] T inicial: {T0:.2f} °C")
            controller.reset(T0)
            for t in clock(duration, dt):
                T_real = lab.T1
                T_ref = referencia_dinamica(t)
                Qu = controller.action(T_real, T_ref)
                Q_nmpc, delta = controller._last
                lab.Q1(Qu)
                log_t.append(t); log_ref.append(T_ref); log_T.append(round(T_real, 3))
                log_Qu.append(Qu); log_Qnmpc.append(Q_nmpc); log_delta.append(delta)
                if t % 30 == 0:
                    print(f"    t={t:4.0f}s ref={T_ref:5.1f} T={T_real:6.2f} "
                          f"Qnmpc={Q_nmpc:5.1f} Δ={delta:6.1f} Qu={Qu:5.1f}%")

    df = pd.DataFrame({
        "Time(s)": log_t, "Reference(C)": log_ref, "Temperature(C)": log_T,
        "Control_Effort(%)": log_Qu, "Base_NMPC(%)": log_Qnmpc,
        "Residual_SAC(%)": log_delta,
    })
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"stress_nmpcsac_rep{rep_idx}.csv")
    df.to_csv(out, index=False)

    e = df["Reference(C)"].values - df["Temperature(C)"].values
    ise = float(np.sum(e**2))
    os_max = float(max(0.0, (df["Temperature(C)"].values - df["Reference(C)"].values).max()))
    energy = float(np.sum(df["Control_Effort(%)"].values))
    tv = float(np.sum(np.abs(np.diff(df["Control_Effort(%)"].values))))
    print(f"  [OK] {out}: ISE={ise:.1f}  OS={os_max:.2f}°C  "
          f"Energía={energy:.0f}  TV={tv:.1f}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Prueba de estrés NMPC-SAC residual con réplicas")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--model", type=str, default="models/nmpc_sac_v2/residual")
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--cooldown", type=int, default=0)
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()

    backend = "sim" if args.sim else "hw"
    print(f"\n{'='*60}")
    print(f"  ESTRÉS NMPC-SAC RESIDUAL | backend={backend} | reps={args.reps}")
    print(f"{'='*60}")

    resultados = []
    for i in range(1, args.reps + 1):
        print(f"\n--- Réplica {i}/{args.reps} ---")
        controller = NMPCSACController(args.model)
        out = ejecutar_una_corrida(controller, backend, args.duration,
                                   args.dt, rep_idx=i, seed=2000 + i, outdir=args.outdir)
        resultados.append(out)
        if args.cooldown > 0 and i < args.reps and backend == "hw":
            print(f"  [..] Enfriando {args.cooldown}s...")
            time.sleep(args.cooldown)

    print(f"\n[LISTO] {len(resultados)} réplicas NMPC-SAC guardadas.")


if __name__ == "__main__":
    main()
