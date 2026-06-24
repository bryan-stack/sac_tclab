"""Entrenamiento del agente SAC con Curriculum Learning para el TCLab.

Entrena un SAC (Stable-Baselines3) sobre el entorno de tclab_env.py. El replay
buffer se pre-llena con trayectorias del PI (IMC-Skogestad) para acelerar el
arranque, y el agente avanza por las cuatro etapas del curriculum según su
recompensa de evaluación. Guarda el mejor modelo y checkpoints periódicos.

Uso:
    python train_sac_v3.py --timesteps 2000000
    python train_sac_v3.py --n-envs 4
    python train_sac_v3.py --resume ./models/sac_tclab_v4_best/best_model
    python train_sac_v3.py --eval-only ./models/sac_tclab_v4_best/best_model
"""

import argparse
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from tclab_env import TCLabEnv

# 1. CONTROLADOR PI PARA PRE-LLENAR EL BUFFER 

class PIController:
    """
    Controlador PI discreto sintonizado con IMC-Skogestad para el FOPDT
    identificado: Kp=0.6052, tau=179.08s, theta=22.19s.

    Reglas IMC-Skogestad con lambda_c = theta (ajuste agresivo):
        Kc   = tau / (Kp * (lambda_c + theta)) = 179.08 / (0.6052 * 44.38) ≈ 6.66
        tauI = min(tau, 4*(lambda_c + theta))  = min(179.08, 177.52) ≈ 177.5s

    Se usa Kc=7.3, tauI=160.3 (ajuste ligeramente más agresivo validado
    en simulación por calibrate_curriculum.py).
    """
    def __init__(self, Kc: float = 7.3, tauI: float = 160.3, dt: float = 1.0):
        self.Kc   = Kc
        self.tauI = tauI
        self.dt   = dt
        self._integral = 0.0

    def reset(self) -> None:
        self._integral = 0.0

    def compute(self, T_ref: float, T_obs: float) -> float:
        """Devuelve Qu ∈ [0,100]%."""
        e = T_ref - T_obs
        self._integral += e * self.dt
        Qu = self.Kc * (e + self._integral / self.tauI)
        return float(np.clip(Qu, 0.0, 100.0))


def prefill_replay_buffer(
    model:       SAC,
    env_fn,
    n_episodes:  int = 300,
    verbose:     bool = True,
) -> int:
    """
    [M1] Pre-llena el replay buffer de SAC con transiciones generadas
    por el controlador PI sintonizado.
    """
    if verbose:
        print(f"\n[M1] Pre-llenando replay buffer con PI controller...")
        print(f"     Episodios: {n_episodes} | "
              f"Buffer capacity: {model.replay_buffer.buffer_size:,} per env")

    pi    = PIController()
    total = 0
    
    # Extraemos cuántos entornos paralelos exige el buffer (ej. 6)
    n_envs = model.env.num_envs
    
    # Lotes temporales para engañar al buffer vectorizado
    b_obs, b_next_obs, b_action, b_reward, b_done, b_infos = [], [], [], [], [], []

    # Crear un env temporal (no vectorizado) para el pre-llenado
    env = env_fn()

    for ep in range(n_episodes):
        obs, _ = env.reset()
        pi.reset()
        done = False

        while not done:
            # Revertir normalización para el PI
            T_mid  = (24.5 + 50.0) / 2.0    # 37.25
            T_half = (50.0 - 24.5) / 2.0    # 12.75
            T_obs_C = float(obs[3]) * T_half + T_mid
            T_ref_C = float(obs[4]) * T_half + T_mid

            Qu = pi.compute(T_ref_C, T_obs_C)
            action_val = float(np.clip(Qu / 50.0 - 1.0, -1.0, 1.0))
            action     = np.array([action_val], dtype=np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Acumular paso en el lote
            b_obs.append(obs)
            b_next_obs.append(next_obs)
            b_action.append(action)
            b_reward.append(reward)
            b_done.append(float(done))
            b_infos.append(info)

            # Cuando el lote tiene el tamaño exigido por el modelo, inyectamos y vaciamos
            if len(b_obs) == n_envs:
                model.replay_buffer.add(
                    obs        = np.vstack(b_obs),
                    next_obs   = np.vstack(b_next_obs),
                    action     = np.vstack(b_action),
                    reward     = np.array(b_reward, dtype=np.float32),
                    done       = np.array(b_done, dtype=np.float32),
                    infos      = b_infos,
                )
                b_obs, b_next_obs, b_action, b_reward, b_done, b_infos = [], [], [], [], [], []

            obs = next_obs
            total += 1

        if verbose and (ep + 1) % 50 == 0:
            print(f"     Episodio {ep+1}/{n_episodes} — total transiciones: {total:,}")

    env.close()

    if verbose:
        print(f"[M1] Buffer pre-llenado: {total:,} transiciones de alta calidad.\n")

    return total


# 2. CALLBACKS

class SyncedCurriculumCallback(BaseCallback):
    """
    Avanza el stage del entorno de entrenamiento cuando la política
    determinista (evaluada por EvalCallback) supera el umbral del stage.

    Lee last_mean_reward directamente desde la referencia al EvalCallback,
    asegurando que el criterio de avance usa la política DETERMINISTA (sin
    ruido de entropía), no el rollout estocástico.
    """
    def __init__(
        self,
        eval_env,
        eval_callback,
        window:  int = 3,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env      = eval_env
        self.eval_callback = eval_callback
        self.window        = window
        self._eval_rewards = []
        self._prev_stage   = 1

    def _on_step(self) -> bool:
        # Sincronizar stage del eval_env con el train_env
        current_stage = self.training_env.get_attr("stage")[0]
        if current_stage != self._prev_stage:
            self.eval_env.env_method("set_stage", current_stage)
            self._prev_stage = current_stage
            if self.verbose:
                print(f"[Curriculum] eval_env → stage {current_stage}")

        # Leer reward determinista del EvalCallback
        current_eval = getattr(self.eval_callback, "last_mean_reward", -np.inf)
        if current_eval > -10000.0:
            if not self._eval_rewards or abs(current_eval - self._eval_rewards[-1]) > 1e-6:
                self._eval_rewards.append(current_eval)
                if len(self._eval_rewards) > self.window:
                    self._eval_rewards.pop(0)

                if current_stage < 4 and len(self._eval_rewards) >= self.window:
                    mean_eval = float(np.mean(self._eval_rewards))
                    thresh    = TCLabEnv.THRESHOLDS_RAW[current_stage]

                    if self.verbose:
                        print(
                            f"\n[Curriculum] Stage {current_stage} | "
                            f"Media eval: {mean_eval:.1f} (meta: {thresh:.1f})"
                        )

                    if mean_eval >= thresh:
                        new_stage = current_stage + 1
                        self.training_env.env_method("set_stage", new_stage)
                        self._eval_rewards.clear()
                        if self.verbose:
                            print(f">>> ¡AVANCE A ETAPA {new_stage}! <<<\n")
        return True


class EntropyAnnealingCallback(BaseCallback):
    """
    Mantiene ent_coef dentro de [floor, ceiling] por stage.

    Problema anterior: solo había piso (floor). Con ent_coef="auto" y
    Stage 3 (dead_time variable + DR), SAC subió ent_coef a 0.303,
    produciendo actor_loss=131 (el gradiente de entropía dominó al de tarea).

    Solución: añadir techo (ceiling) ~5x el floor. El optimizador tiene
    margen para subir ent_coef si lo necesita, pero no puede desbocarse.
    """
    STAGE_FLOORS   = {1: 0.010, 2: 0.005, 3: 0.002, 4: 0.001}
    STAGE_CEILINGS = {1: 0.050, 2: 0.025, 3: 0.010, 4: 0.005}

    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        current_stage = self.training_env.get_attr("stage")[0]
        self.logger.record("curriculum/stage", current_stage)

        if not (hasattr(self.model, "log_ent_coef") and
                self.model.log_ent_coef is not None):
            return True

        floor   = self.STAGE_FLOORS.get(current_stage, 0.001)
        ceiling = self.STAGE_CEILINGS.get(current_stage, 0.05)

        with torch.no_grad():
            current_ent = float(self.model.log_ent_coef.exp())

        if current_ent < floor:
            self.model.log_ent_coef.data.fill_(math.log(floor))
            if self.verbose:
                print(f"[Entropy] Stage {current_stage}: "
                      f"{current_ent:.5f} < floor={floor} → {floor}")
        elif current_ent > ceiling:
            self.model.log_ent_coef.data.fill_(math.log(ceiling))
            if self.verbose:
                print(f"[Entropy] Stage {current_stage}: "
                      f"{current_ent:.4f} > ceiling={ceiling} → recortado")
        return True


class MetricsCallback(BaseCallback):
    """Loguea reward_raw y stage al TensorBoard cada log_freq pasos."""
    def __init__(self, log_freq: int = 1000):
        super().__init__(verbose=0)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            try:
                stage = self.training_env.get_attr("stage")[0]
                self.logger.record("custom/stage", stage)
            except Exception:
                pass
        return True


# 3. DETECCIÓN DE HARDWARE Y CONFIGURACIÓN

def detect_hardware(n_envs_override=None):
    """
    Detecta el hardware y devuelve (device, n_envs).

    Apple Silicon: CPU + 1 env (SubprocVecEnv tiene overhead alto en macOS)
    x86 (i5/i7/Ryzen): CPU + min(n_cores-2, 6) envs en SubprocVecEnv
    GPU (CUDA): CPU igualmente — red [64,64] + batch=256 no amortiza PCIe
    """
    n_cores  = os.cpu_count() or 4
    is_apple = platform.system() == "Darwin" and platform.machine() == "arm64"

    if is_apple:
        n_envs = n_envs_override or 1
        torch.set_num_threads(min(6, n_cores - 2))
        print(f"[HW] Apple Silicon — CPU, {n_envs} env")
    else:
        if torch.cuda.is_available():
            print(f"[HW] GPU {torch.cuda.get_device_name(0)} detectada — "
                  "usando CPU + SubprocVecEnv (más rápido para esta arquitectura)")
        n_envs = n_envs_override or min(max(1, n_cores - 2), 6)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        print(f"[HW] CPU x86 — {n_cores} cores, {n_envs} envs")

    return "cpu", n_envs


# 4. CONSTRUCCIÓN DE ENTORNOS

def make_env(stage: int, seed: int):
    def _init():
        env = TCLabEnv(stage=stage, seed=seed)
        return Monitor(env)
    return _init


# 5. ENTRENAMIENTO

def train(
    total_timesteps:  int  = 2_000_000,
    n_envs_override:  int  = None,
    resume_path:      str  = None,
    log_dir:          str  = "./logs/sac_tclab_v4",
    save_path:        str  = "./models/sac_tclab_v4",
    seed:             int  = 42,
    prefill_episodes: int  = 300,
    eval_freq_steps:  int  = 10_000,
    checkpoint_steps: int  = 25_000,
) -> SAC:

    device, n_envs = detect_hardware(n_envs_override)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SAC TCLab v4 — Entrenamiento")
    print(f"  Timesteps    : {total_timesteps:,}")
    print(f"  N_ENVS       : {n_envs}")
    print(f"  Obs. size    : {TCLabEnv.OBS_SIZE}  (5+1+25+15)")
    print(f"  Pre-fill PI  : {prefill_episodes} episodios")
    print(f"  Mejoras      : M1(PI-buffer) M2(500s) M3(cold-start) "
          f"M4(Ta-obs) M5(anti-OS)")
    print(f"{'='*60}\n")

    # Entornos
    train_fns = [make_env(stage=1, seed=seed + i) for i in range(n_envs)]
    eval_fns  = [make_env(stage=1, seed=seed + 1000)]

    train_vec = SubprocVecEnv(train_fns) if n_envs > 1 else DummyVecEnv(train_fns)
    eval_vec  = DummyVecEnv(eval_fns)

    # Modelo SAC
    # target_entropy=-1.0 es el valor estándar para espacio de acción 1D.
    # Con obs_size=46, la red [64,64] tiene ~46*64 + 64*64 + 64*1 ≈ 7100
    # parámetros en el actor — suficiente para la tarea y rápido de inferir.
    sac_kwargs = dict(
        policy              = "MlpPolicy",
        env                 = train_vec,
        learning_rate       = 3e-4,
        buffer_size         = 300_000,
        batch_size          = 256,
        tau                 = 0.005,
        gamma               = 0.99,
        ent_coef            = "auto",
        target_entropy      = -1.0,
        train_freq          = 1,
        gradient_steps      = 1,
        learning_starts     = 1_000,   # bajo porque el buffer viene pre-llenado
        policy_kwargs       = dict(net_arch=[64, 64]),
        tensorboard_log     = log_dir,
        device              = device,
        verbose             = 1,
        seed                = seed,
    )

    if resume_path and Path(resume_path + ".zip").exists():
        print(f"[RESUME] Cargando desde {resume_path}")
        model = SAC.load(
            resume_path,
            env=train_vec,
            device=device,
            custom_objects={"learning_rate": 3e-4},
        )
        # No pre-llenar al resumir — el buffer ya tiene experiencia útil
        prefill_episodes = 0
    else:
        model = SAC(**sac_kwargs)

    # Pre-llenar el replay buffer con trayectorias PI
    if prefill_episodes > 0:
        env_fn_for_prefill = make_env(stage=1, seed=seed + 9999)
        prefill_replay_buffer(
            model         = model,
            env_fn        = env_fn_for_prefill,
            n_episodes    = prefill_episodes,
            verbose       = True,
        )

    # Callbacks
    eval_cb = EvalCallback(
        eval_vec,
        best_model_save_path = f"{save_path}_best",
        log_path             = log_dir,
        eval_freq            = max(eval_freq_steps  // n_envs, 1),
        n_eval_episodes      = 15,
        deterministic        = True,
        verbose              = 0,
    )

    curriculum_cb = SyncedCurriculumCallback(
        eval_env      = eval_vec,
        eval_callback = eval_cb,
        window        = 3,
        verbose       = 1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq   = max(checkpoint_steps // n_envs, 1),
        save_path   = str(Path(save_path).parent / "checkpoints"),
        name_prefix = "sac_tclab_v4",
        verbose     = 0,
    )

    entropy_cb = EntropyAnnealingCallback(check_freq=1000)
    metrics_cb = MetricsCallback(log_freq=1000)

    # Entrenamiento
    t0 = time.time()
    try:
        model.learn(
            total_timesteps    = total_timesteps,
            callback           = [eval_cb, curriculum_cb, checkpoint_cb,
                                  entropy_cb, metrics_cb],
            reset_num_timesteps = (resume_path is None),
            progress_bar       = False,
        )
    except KeyboardInterrupt:
        print("\n[!] Interrumpido por usuario — guardando modelo actual...")

    elapsed = time.time() - t0
    model.save(save_path)
    final_stage = train_vec.get_attr("stage")[0]

    print(f"\n[OK] Modelo guardado en {save_path}.zip")
    print(f"[OK] Tiempo total  : {elapsed/3600:.2f} h")
    print(f"[OK] Stage final   : {final_stage}")
    print(f"[OK] TensorBoard   : tensorboard --logdir {log_dir}")

    train_vec.close()
    eval_vec.close()
    return model


# 6. EVALUACIÓN POST-ENTRENAMIENTO

def evaluate(
    model_path:  str,
    n_episodes:  int = 20,
    stage:       int = 4,
    seed:        int = 0,
) -> dict:
    """
    Evalúa el modelo con política determinista. Calcula ISE/ITSE/IAE/ITAE.
    Comparar con baseline NMPC-TD3 del paper de referencia.
    """
    env   = Monitor(TCLabEnv(stage=stage, seed=seed))
    model = SAC.load(model_path, env=env, device="cpu")

    metrics = {k: [] for k in ["ISE", "ITSE", "IAE", "ITAE", "reward"]}

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ise = itse = iae = itae = ep_reward = t_s = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            t_s       += env.unwrapped.dt
            eT_abs     = abs(info.get("eT", 0.0))
            ise   += eT_abs ** 2
            itse  += t_s * eT_abs ** 2
            iae   += eT_abs
            itae  += t_s * eT_abs
            done = terminated or truncated

        metrics["ISE"].append(ise)
        metrics["ITSE"].append(itse)
        metrics["IAE"].append(iae)
        metrics["ITAE"].append(itae)
        metrics["reward"].append(ep_reward)

    baseline = {
        1: {"ISE": 320.10, "IAE": 162.00},
        2: {"ISE":  75.41, "IAE": 178.40},
        3: {"ISE":  71.07, "IAE":  58.88},
        4: {"ISE":  71.07, "IAE":  58.88},  # mismo test 3 como referencia
    }

    print(f"\n── Evaluación stage {stage}  ({n_episodes} episodios) ──")
    print(f"{'Métrica':8} {'Media':>12} {'Std':>10} {'NMPC-TD3':>10} {'Ratio':>8}")
    print("─" * 52)
    for k in ["ISE", "ITSE", "IAE", "ITAE"]:
        mean = np.mean(metrics[k])
        std  = np.std(metrics[k])
        bval = baseline.get(stage, {}).get(k, None)
        if bval:
            ratio_str = f"{mean/bval:.2f}x"
            flag      = "✓" if mean < bval else "·"
        else:
            ratio_str = "—"
            flag      = ""
        print(f"{k:8} {mean:>12.3f} {std:>10.3f} {str(bval or '—'):>10} "
              f"{ratio_str:>8} {flag}")
    print(f"{'reward':8} {np.mean(metrics['reward']):>12.3f}")
    print()

    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items()}


# 7. CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAC TCLab v4 — entrenar / evaluar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python train_sac_v3.py                               # entrenar desde cero
  python train_sac_v3.py --timesteps 3000000
  python train_sac_v3.py --n-envs 4                    # forzar 4 envs
  python train_sac_v3.py --resume ./models/checkpoints/sac_tclab_v4_500000_steps
  python train_sac_v3.py --eval-only ./models/sac_tclab_v4_best/best_model --stage 4
        """,
    )
    parser.add_argument("--timesteps",        type=int,  default=2_000_000)
    parser.add_argument("--n-envs",           type=int,  default=None)
    parser.add_argument("--seed",             type=int,  default=42)
    parser.add_argument("--resume",           type=str,  default=None)
    parser.add_argument("--eval-only",        type=str,  default=None)
    parser.add_argument("--stage",            type=int,  default=4)
    parser.add_argument("--eval-episodes",    type=int,  default=20)
    parser.add_argument("--prefill-episodes", type=int,  default=300,
                        help="Episodios PI para pre-llenar buffer (0=desactivar)")
    parser.add_argument("--log-dir",          type=str,  default="./logs/sac_tclab_v4")
    parser.add_argument("--save-path",        type=str,  default="./models/sac_tclab_v4")
    args = parser.parse_args()

    if args.eval_only:
        evaluate(args.eval_only, n_episodes=args.eval_episodes, stage=args.stage)
    else:
        train(
            total_timesteps  = args.timesteps,
            n_envs_override  = args.n_envs,
            resume_path      = args.resume,
            log_dir          = args.log_dir,
            save_path        = args.save_path,
            seed             = args.seed,
            prefill_episodes = args.prefill_episodes,
        )