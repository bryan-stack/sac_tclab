"""Prueba de estrés multi-controlador con réplicas sobre el TCLab.

Ejecuta el perfil de estrés dinámico de 600 s para el PI (--mode pi), el NMPC
puro (--mode nmpc) o el SAC puro (--mode sac), con --reps réplicas para obtener
media y desviación estándar. Por defecto corre en el hardware real (librería
tclab); con --sim usa el modelo ODE no lineal, con tiempo muerto y ruido de ADC,
para validar sin hardware. El NMPC-SAC residual se ejecuta aparte en
nmpc/stress_nmpcsac.py, por usar un espacio de observación distinto.

Cada corrida guarda stress_<mode>_rep<i>.csv con columnas
Time(s), Reference(C), Temperature(C), Control_Effort(%). Ejemplos:

    python stress_suite_v2.py --mode pi   --reps 4
    python stress_suite_v2.py --mode nmpc --reps 4 --Tp 30
    python stress_suite_v2.py --mode sac  --reps 4 --model models/sac_tclab_v4_best/best_model
"""

import os
import argparse
import importlib.util
import time
import numpy as np
import pandas as pd

# Entorno TCLab de trainv2 (46 dim). Se resuelve desde el directorio del
# script; NO se modifica sys.path para no colisionar con nmpc/tclab_env.py.
from tclab_env import TCLabEnv, nonlinear_thermal_ode, NOMINAL_PARAMS  # trainv2


def _load_nmpc_solver():
    """Carga NMPCSolver desde ../nmpc/nmpc_solver.py de forma aislada
    (sin tocar sys.path, para no colisionar con el tclab_env de nmpc/)."""
    nmpc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nmpc")
    path = os.path.join(nmpc_dir, "nmpc_solver.py")
    spec = importlib.util.spec_from_file_location("nmpc_solver_ext", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NMPCSolver

# 1. PERFIL DINÁMICO DE ESTRÉS (idéntico al de stress_test_suite.py)
def referencia_dinamica(t: float) -> float:
    if t < 150:
        return 35.0
    elif t < 300:
        return 48.0
    elif t < 450:
        return 28.0
    else:
        return 40.0


# 2. PLANTA SIMULADA (solo para --sim; reproduce dead-time y ruido ADC)
class PlantSim:
    """Planta TCLab simulada: ODE no lineal + dead-time + ruido de cuantización.

    Se usa exclusivamente para validar los scripts sin hardware. Aplica
    Domain Randomization leve (±5 %) sobre Uh y K para que el NMPC (que usa
    parámetros nominales) enfrente una planta ligeramente distinta, como en
    la realidad.
    """
    def __init__(self, T0_C=32.0, Ta_C=32.0, dead_steps=22, noise_std=0.10,
                 dt=1.0, seed=None, dr=0.05):
        self.dt = dt
        self.noise_std = noise_std
        self.dead_steps = dead_steps
        self.rng = np.random.default_rng(seed)
        self.params = dict(NOMINAL_PARAMS)
        self.params["Ta"] = 273.15 + Ta_C
        if dr > 0:
            self.params["Uh"] *= self.rng.uniform(1 - dr, 1 + dr)
            self.params["K"]  *= self.rng.uniform(1 - dr, 1 + dr)
        self.T_K = T0_C + 273.15
        self._buffer = [T0_C] * (dead_steps + 1)

    def step(self, Qu: float) -> float:
        """Aplica Qu durante dt y devuelve la temperatura observada (retardada)."""
        dTdt = nonlinear_thermal_ode(self.T_K, float(np.clip(Qu, 0, 100)), self.params)
        self.T_K += dTdt * self.dt
        T_real = self.T_K - 273.15 + self.rng.normal(0, self.noise_std)
        self._buffer.append(T_real)
        return self._buffer.pop(0) if self.dead_steps > 0 else T_real

    @property
    def T1(self) -> float:
        """Lectura instantánea (observada con dead-time) sin avanzar la planta."""
        return self._buffer[0]


# 3. CONTROLADORES (interfaz común: reset(T0) y action(T_real, T_ref) -> Qu)
class PIController:
    """PI clásico con anti-windup condicional (IMC-Skogestad)."""
    def __init__(self, Kc=7.3, tauI=160.3, dt=1.0):
        self.Kc, self.tauI, self.dt = Kc, tauI, dt
        self._integral = 0.0

    def reset(self, T0_C):
        self._integral = 0.0

    def action(self, T_real, T_ref):
        e = T_ref - T_real
        self._integral += e * self.dt
        Qu = self.Kc * (e + self._integral / self.tauI)
        if Qu > 100.0:
            Qu = 100.0
            self._integral -= e * self.dt   # anti-windup condicional
        elif Qu < 0.0:
            Qu = 0.0
            self._integral -= e * self.dt
        return Qu


class NMPCController:
    """NMPC puro. Tp configurable; por defecto Tp=30 > theta=22 (corrige M1.2).

    Usa parámetros NOMINALES (no conoce el DR real de la planta), como en
    un despliegue realista. Recalcula en cada paso (sin cacheo) para ser el
    benchmark predictivo más justo posible.
    """
    def __init__(self, Tp=30, Tc=5, dt=1.0):
        NMPCSolver = _load_nmpc_solver()
        self.solver = NMPCSolver(dt=dt, Tp=Tp, Tc=Tc)
        self.Q_prev = 30.0

    def reset(self, T0_C):
        self.Q_prev = 30.0
        # Sincroniza Ta del modelo interno con la temperatura inicial medida
        p = dict(NOMINAL_PARAMS)
        p["Ta"] = 273.15 + T0_C
        self.solver.update_params(**p)

    def action(self, T_real, T_ref):
        Q = self.solver.compute_action(T0_C=T_real, T_ref_C=T_ref, Q_warm=self.Q_prev)
        self.Q_prev = Q
        return float(np.clip(Q, 0.0, 100.0))


class SACController:
    """SAC puro (model-free). Construye la observación de 46 dim con un
    TCLabEnv (trainv2) interno, replicando SACHardwareAdapter."""
    def __init__(self, model_path, dt=1.0):
        from stable_baselines3 import SAC
        self.model = SAC.load(model_path, device="cpu")
        self.env = TCLabEnv(stage=4)
        self.dt = dt

    def reset(self, T0_C):
        self.env.reset()
        self.env.T_K = T0_C + 273.15
        self.env.Ta_C = T0_C
        self.env.eT_prev = 0.0
        self.env.eT_integral = 0.0
        self.env._action_history = [0.0] * self.env.HISTORY_LEN
        self.env.step_count = 0
        self._obs = self.env._build_obs(T0_C, 0.0, 0.0)

    def action(self, T_real, T_ref):
        act, _ = self.model.predict(self._obs, deterministic=True)
        self.env.T_ref_C = T_ref
        Qu = float(np.clip((act[0] + 1.0) * 50.0, 0.0, 100.0))
        self.env._action_history.append(float(act[0]))
        self.env._action_history.pop(0)
        eT_real = T_ref - T_real
        eT_der = (eT_real - self.env.eT_prev) / self.dt
        self.env.eT_integral += eT_real * self.dt
        self.env.eT_prev = eT_real
        self.env.step_count += 1
        self._obs = self.env._build_obs(T_real, eT_real, eT_der)
        return Qu


# 4. MOTOR DE EJECUCIÓN (hardware o simulación)
def ejecutar_una_corrida(mode, controller, backend, duration, dt, rep_idx, seed, outdir="."):
    log_t, log_ref, log_T, log_Qu = [], [], [], []

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
            plant.step(Qu)             # avanza la planta con la acción
            log_t.append(t); log_ref.append(T_ref)
            log_T.append(round(T_real, 3)); log_Qu.append(Qu)
    else:
        import tclab
        from tclab import clock
        with tclab.TCLab() as lab:
            T0 = lab.T1
            print(f"  [+] T inicial: {T0:.2f} °C (verifica que esté ~30-33 °C)")
            controller.reset(T0)
            for t in clock(duration, dt):
                T_real = lab.T1
                T_ref = referencia_dinamica(t)
                Qu = controller.action(T_real, T_ref)
                lab.Q1(Qu)
                log_t.append(t); log_ref.append(T_ref)
                log_T.append(round(T_real, 3)); log_Qu.append(Qu)
                if t % 30 == 0:
                    print(f"    t={t:4.0f}s  ref={T_ref:5.1f}  T={T_real:6.2f}  Qu={Qu:5.1f}%")

    df = pd.DataFrame({
        "Time(s)": log_t, "Reference(C)": log_ref,
        "Temperature(C)": log_T, "Control_Effort(%)": log_Qu,
    })
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"stress_{mode}_rep{rep_idx}.csv")
    df.to_csv(out, index=False)

    # Métricas rápidas para feedback inmediato
    e = df["Reference(C)"].values - df["Temperature(C)"].values
    ise = float(np.sum(e**2))
    os_max = float(max(0.0, (df["Temperature(C)"].values - df["Reference(C)"].values).max()))
    energy = float(np.sum(df["Control_Effort(%)"].values))
    tv = float(np.sum(np.abs(np.diff(df["Control_Effort(%)"].values))))
    print(f"  [OK] {out}: ISE={ise:.1f}  OS={os_max:.2f}°C  "
          f"Energía={energy:.0f}  TV={tv:.1f}")
    return out


def construir_controlador(mode, model_path, Tp):
    if mode == "pi":
        return PIController()
    if mode == "nmpc":
        return NMPCController(Tp=Tp, Tc=5)
    if mode == "sac":
        if not model_path:
            model_path = "models/sac_tclab_v4_best/best_model"
        return SACController(model_path)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser(description="Suite de estrés multi-controlador con réplicas")
    ap.add_argument("--mode", required=True, choices=["pi", "nmpc", "sac"])
    ap.add_argument("--reps", type=int, default=4, help="Número de réplicas")
    ap.add_argument("--model", type=str, default=None, help="Ruta al modelo SAC")
    ap.add_argument("--Tp", type=int, default=30, help="Horizonte de predicción NMPC (>=theta=22)")
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--sim", action="store_true", help="Backend de simulación (sin hardware)")
    ap.add_argument("--cooldown", type=int, default=0,
                    help="Segundos de pausa entre réplicas (hardware: dejar enfriar)")
    ap.add_argument("--outdir", type=str, default=".",
                    help="Directorio de salida (usa 'sim_runs' para validación)")
    args = ap.parse_args()

    backend = "sim" if args.sim else "hw"
    print(f"\n{'='*60}")
    print(f"  SUITE DE ESTRÉS — modo={args.mode.upper()} | backend={backend} | reps={args.reps}")
    if args.mode == "nmpc":
        print(f"  NMPC: Tp={args.Tp}s (theta=22s -> Tp>theta OK)")
    print(f"{'='*60}")

    resultados = []
    for i in range(1, args.reps + 1):
        print(f"\n--- Réplica {i}/{args.reps} ---")
        # Controlador nuevo por réplica (estado limpio); semilla distinta en sim
        controller = construir_controlador(args.mode, args.model, args.Tp)
        out = ejecutar_una_corrida(
            args.mode, controller, backend,
            args.duration, args.dt, rep_idx=i, seed=1000 + i, outdir=args.outdir,
        )
        resultados.append(out)
        if args.cooldown > 0 and i < args.reps and backend == "hw":
            print(f"  [..] Enfriando {args.cooldown}s antes de la siguiente réplica...")
            time.sleep(args.cooldown)

    print(f"\n[LISTO] {len(resultados)} réplicas guardadas para modo '{args.mode}':")
    for r in resultados:
        print(f"        {r}")


if __name__ == "__main__":
    main()
