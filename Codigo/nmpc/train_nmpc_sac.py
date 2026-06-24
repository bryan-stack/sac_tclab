"""Entrenamiento de la corrección residual NMPC-SAC para el TCLab.

Entrena un SAC que aprende un residuo Δ_SAC sobre la acción del NMPC
(Qu = clip(Q_NMPC + Δ_SAC, 0, 100)). La red es pequeña ([128, 128]) porque el
residuo es una señal acotada, y el buffer se pre-llena con Δ_SAC = 0 para que
el agente parta de las trayectorias del NMPC puro y aprenda a corregirlas.
"""

import argparse
import math
import os
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


# 1. PREFILL CON DELTA=0 (TRAYECTORIAS DEL NMPC PURO)

def prefill_replay_buffer(
    model:      SAC,
    env_fn,                       # callable(stage) → gym.Env
    n_episodes: int  = 200,
    verbose:    bool = True,
) -> int:
    """
    [F7] Pre-llena el replay buffer con transiciones donde Δ_SAC = 0.
    El entorno calcula Qu = Q_NMPC + 0 = Q_NMPC (NMPC puro).
    SAC ve en el buffer: "con esta política base, así son las trayectorias".
    Luego aprende a desviarse de Δ=0 cuando ese desvío mejora el reward.

    Distribución:
      - 50% episodios Stage 1 (regulación)
      - 50% episodios Stage 2 (tracking con escalones)
    """
    if verbose:
        print(f"\n[Prefill] Generando {n_episodes} episodios con NMPC puro (Δ=0)...")
        print(f"          50% Stage 1, 50% Stage 2")

    n_envs = model.env.num_envs
    total  = 0
    b_obs, b_next, b_act, b_rew, b_done, b_inf = [], [], [], [], [], []

    env_s1 = env_fn(stage=1)
    env_s2 = env_fn(stage=2)

    # Acción neutra: en espacio normalizado [-1,1], el 0 = sin corrección
    action_zero = np.array([0.0], dtype=np.float32)

    for ep in range(n_episodes):
        env = env_s1 if ep < n_episodes // 2 else env_s2
        obs, _ = env.reset()
        done = False

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action_zero)
            done = terminated or truncated

            b_obs.append(obs)
            b_next.append(next_obs)
            b_act.append(action_zero)
            b_rew.append(reward)
            b_done.append(float(done))
            b_inf.append({})

            if len(b_obs) == n_envs:
                model.replay_buffer.add(
                    obs      = np.vstack(b_obs),
                    next_obs = np.vstack(b_next),
                    action   = np.vstack(b_act),
                    reward   = np.array(b_rew, dtype=np.float32),
                    done     = np.array(b_done, dtype=np.float32),
                    infos    = b_inf,
                )
                b_obs, b_next, b_act, b_rew, b_done, b_inf = [], [], [], [], [], []

            obs = next_obs
            total += 1

        if verbose and (ep + 1) % 25 == 0:
            tag = "S1" if ep < n_episodes // 2 else "S2"
            print(f"          ep {ep+1:>3}/{n_episodes} [{tag}] — transiciones: {total:,}")

    env_s1.close()
    env_s2.close()
    if verbose:
        print(f"[Prefill] Completado: {total:,} transiciones del NMPC puro.\n")
    return total


# 2. CALLBACKS

class SyncedCurriculumCallback(BaseCallback):
    """Avanza el stage cuando la política determinista supera el umbral."""
    def __init__(self, eval_env, eval_callback, window=3, verbose=1):
        super().__init__(verbose)
        self.eval_env      = eval_env
        self.eval_callback = eval_callback
        self.window        = window
        self._eval_rewards = []
        self._prev_stage   = 1

    def _on_step(self) -> bool:
        current_stage = self.training_env.get_attr("stage")[0]
        if current_stage != self._prev_stage:
            self.eval_env.env_method("set_stage", current_stage)
            self._prev_stage = current_stage
            if self.verbose:
                print(f"[Curriculum] eval_env → stage {current_stage}")

        current_eval = getattr(self.eval_callback, "last_mean_reward", -np.inf)
        if current_eval > -10000.0:
            if not self._eval_rewards or abs(current_eval - self._eval_rewards[-1]) > 1e-6:
                self._eval_rewards.append(current_eval)
                if len(self._eval_rewards) > self.window:
                    self._eval_rewards.pop(0)

                if current_stage < 4 and len(self._eval_rewards) >= self.window:
                    mean_eval = float(np.mean(self._eval_rewards))
                    thresh    = TCLabEnv.THRESHOLDS_RAW[current_stage]
                    if mean_eval >= thresh:
                        new_stage = current_stage + 1
                        self.training_env.env_method("set_stage", new_stage)
                        self._eval_rewards.clear()
                        if self.verbose:
                            print(f"\n>>> AVANCE A ETAPA {new_stage} "
                                  f"(media={mean_eval:.1f} ≥ umbral={thresh:.1f}) <<<\n")
        return True


class BestModelPerStageCallback(BaseCallback):
    """
    [F8] Guarda el mejor modelo POR STAGE para evitar el bug del best_model
    congelado en Stage 1 (problema observado en runs anteriores).
    """
    def __init__(self, eval_callback, save_dir, verbose=1):
        super().__init__(verbose)
        self.eval_cb       = eval_callback
        self.save_dir      = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.current_stage = 1
        self.best_reward   = -np.inf
        self._last_save_eval = None

    def _on_step(self) -> bool:
        stage = self.training_env.get_attr("stage")[0]
        if stage != self.current_stage:
            self.current_stage = stage
            self.best_reward   = -np.inf
            self._last_save_eval = None
            if self.verbose:
                print(f"[BestPerStage] Reseteado para stage {stage}")

        if hasattr(self.eval_cb, "last_mean_reward"):
            res = self.eval_cb.last_mean_reward
            # Sólo guardar si es una NUEVA evaluación (no el mismo valor repetido)
            if res > -10000.0 and res != self._last_save_eval and res > self.best_reward:
                self.best_reward = res
                self._last_save_eval = res
                path = self.save_dir / f"best_model_stage_{stage}"
                self.model.save(str(path))
                if self.verbose:
                    print(f"[BestPerStage] Stage {stage}: nuevo mejor reward = {res:.2f}")
        return True


class EntropyAnnealingCallback(BaseCallback):
    """Mantiene ent_coef en [floor, ceiling] por stage (anti-runaway)."""
    STAGE_FLOORS   = {1: 0.010, 2: 0.005, 3: 0.002, 4: 0.001}
    STAGE_CEILINGS = {1: 0.050, 2: 0.025, 3: 0.010, 4: 0.005}

    def __init__(self, check_freq=1000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
        stage = self.training_env.get_attr("stage")[0]
        self.logger.record("curriculum/stage", stage)

        if not (hasattr(self.model, "log_ent_coef") and self.model.log_ent_coef is not None):
            return True

        floor   = self.STAGE_FLOORS.get(stage, 0.001)
        ceiling = self.STAGE_CEILINGS.get(stage, 0.05)
        with torch.no_grad():
            current = float(self.model.log_ent_coef.exp())
        if current < floor:
            self.model.log_ent_coef.data.fill_(math.log(floor))
        elif current > ceiling:
            self.model.log_ent_coef.data.fill_(math.log(ceiling))
        return True


# 3. ENTRENAMIENTO

def detect_hardware(n_envs_override=None):
    n_cores = os.cpu_count() or 4
    if torch.cuda.is_available():
        print(f"[HW] GPU {torch.cuda.get_device_name(0)} detectada — "
              f"usando CPU + SubprocVecEnv (mejor para esta red)")
    n_envs = n_envs_override or min(max(1, n_cores - 2), 6)
    torch.set_num_threads(2)
    return "cpu", n_envs


def make_env(stage: int, seed: int):
    def _init():
        env = TCLabEnv(stage=stage, seed=seed)
        return Monitor(env)
    return _init


def train(
    total_timesteps:  int  = 1_500_000,
    n_envs_override:  int  = None,
    prefill_episodes: int  = 200,
    log_dir:          str  = "./logs/nmpc_sac_v2",
    save_path:        str  = "./models/nmpc_sac_v2/residual",
    seed:             int  = 42,
) -> SAC:

    device, n_envs = detect_hardware(n_envs_override)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  NMPC-SAC RESIDUAL — entrenamiento")
    print(f"  Timesteps         : {total_timesteps:,}")
    print(f"  N_ENVS            : {n_envs}")
    print(f"  Obs size          : {TCLabEnv.OBS_SIZE}")
    print(f"  Action            : Δ_SAC ∈ [-{TCLabEnv.MAX_RESIDUAL}, "
          f"+{TCLabEnv.MAX_RESIDUAL}] %")
    print(f"  NMPC refresh      : cada {TCLabEnv.NMPC_REFRESH_STEPS} pasos")
    print(f"  Prefill (NMPC Δ=0): {prefill_episodes} episodios")
    print(f"{'='*64}\n")

    train_fns = [make_env(stage=1, seed=seed + i) for i in range(n_envs)]
    eval_fns  = [make_env(stage=1, seed=seed + 1000)]
    train_vec = SubprocVecEnv(train_fns) if n_envs > 1 else DummyVecEnv(train_fns)
    eval_vec  = DummyVecEnv(eval_fns)

    # Red más pequeña — el residuo no requiere [256,256]
    sac_kwargs = dict(
        policy          = "MlpPolicy",
        env             = train_vec,
        learning_rate   = 1e-4,
        buffer_size     = 300_000,
        batch_size      = 256,
        tau             = 0.005,
        gamma           = 0.99,
        ent_coef        = "auto",
        target_entropy  = -1.0,
        learning_starts = 2_000,
        policy_kwargs   = dict(net_arch=[128, 128]),
        tensorboard_log = log_dir,
        device          = device,
        verbose         = 1,
        seed            = seed,
    )
    model = SAC(**sac_kwargs)

    # Prefill con NMPC puro (Δ=0)
    if prefill_episodes > 0:
        def env_factory(stage):
            return make_env(stage=stage, seed=seed + 9999)()
        prefill_replay_buffer(
            model      = model,
            env_fn     = env_factory,
            n_episodes = prefill_episodes,
            verbose    = True,
        )

    eval_cb = EvalCallback(
        eval_vec,
        best_model_save_path = f"{save_path}_best_global",
        log_path             = log_dir,
        eval_freq            = max(10_000 // n_envs, 1),
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 0,
    )
    curriculum_cb = SyncedCurriculumCallback(eval_vec, eval_cb)
    best_stage_cb = BestModelPerStageCallback(
        eval_cb, str(Path(save_path).parent / "best_per_stage"))
    entropy_cb    = EntropyAnnealingCallback()
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(50_000 // n_envs, 1),
        save_path   = str(Path(save_path).parent / "checkpoints"),
        name_prefix = "nmpc_sac",
        verbose     = 0,
    )

    t0 = time.time()
    try:
        model.learn(
            total_timesteps = total_timesteps,
            callback        = [eval_cb, curriculum_cb, best_stage_cb,
                               entropy_cb, checkpoint_cb],
            progress_bar    = False,
        )
    except KeyboardInterrupt:
        print("\n[!] Interrumpido — guardando modelo actual...")

    elapsed = time.time() - t0
    model.save(save_path)
    final_stage = train_vec.get_attr("stage")[0]

    print(f"\n[OK] Modelo guardado en {save_path}.zip")
    print(f"[OK] Tiempo total : {elapsed/3600:.2f} h")
    print(f"[OK] Stage final  : {final_stage}")
    print(f"[OK] TensorBoard  : tensorboard --logdir {log_dir}")
    print(f"[OK] Best per stage en {Path(save_path).parent / 'best_per_stage'}/")

    train_vec.close()
    eval_vec.close()
    return model


# 4. EVALUACIÓN

def evaluate(model_path: str, stage: int = 4, n_episodes: int = 20, seed: int = 0):
    env   = Monitor(TCLabEnv(stage=stage, seed=seed))
    model = SAC.load(model_path, env=env, device="cpu")

    baseline = {
        1: {"ISE": 320.10, "IAE": 162.00},
        2: {"ISE":  75.41, "IAE": 178.40},
        3: {"ISE":  71.07, "IAE":  58.88},
        4: {"ISE":  71.07, "IAE":  58.88},
    }

    metrics = {k: [] for k in ["ISE", "ITSE", "IAE", "ITAE", "reward",
                                "abs_delta_Q", "Qu_max"]}

    print(f"\n[Eval Stage {stage}]  modelo = {model_path}")
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ise = itse = iae = itae = ep_r = t_s = 0.0
        sum_delta = 0.0; n = 0; Qu_max = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_r += reward
            t_s  += env.unwrapped.dt
            eT_a  = abs(info["eT"])
            ise  += eT_a ** 2
            itse += t_s * eT_a ** 2
            iae  += eT_a
            itae += t_s * eT_a
            sum_delta += abs(info["delta_Q"])
            n += 1
            Qu_max = max(Qu_max, info["Qu"])
            done = term or trunc
        metrics["ISE"].append(ise)
        metrics["ITSE"].append(itse)
        metrics["IAE"].append(iae)
        metrics["ITAE"].append(itae)
        metrics["reward"].append(ep_r)
        metrics["abs_delta_Q"].append(sum_delta / max(n, 1))
        metrics["Qu_max"].append(Qu_max)

    print(f"\n── Resultados ({n_episodes} episodios) ──")
    print(f"{'Métrica':12} {'Media':>12} {'Std':>10} {'NMPC-TD3':>10} {'Ratio':>8}")
    print("─" * 56)
    for k in ["ISE", "ITSE", "IAE", "ITAE"]:
        m = float(np.mean(metrics[k]))
        s = float(np.std(metrics[k]))
        b = baseline.get(stage, {}).get(k, None)
        if b is not None:
            r   = m / b
            flag = "✓" if r < 1.0 else "·"
            print(f"{k:12} {m:>12.2f} {s:>10.2f} {b:>10.2f} {r:>7.2f}x {flag}")
        else:
            print(f"{k:12} {m:>12.2f} {s:>10.2f} {'—':>10} {'—':>8}")
    print(f"{'reward':12} {float(np.mean(metrics['reward'])):>12.2f}")
    print(f"{'|Δ_Q| medio':12} {float(np.mean(metrics['abs_delta_Q'])):>12.2f} %  "
          f"(menor = más cerca del NMPC puro)")
    print(f"{'Qu_max':12} {float(np.mean(metrics['Qu_max'])):>12.2f} %")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NMPC-SAC residual para TCLab")
    parser.add_argument("--timesteps",        type=int, default=1_500_000)
    parser.add_argument("--n-envs",           type=int, default=None)
    parser.add_argument("--prefill-episodes", type=int, default=200)
    parser.add_argument("--eval-only",        type=str, default=None)
    parser.add_argument("--stage",            type=int, default=4)
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()

    if args.eval_only:
        evaluate(args.eval_only, stage=args.stage)
    else:
        train(
            total_timesteps  = args.timesteps,
            n_envs_override  = args.n_envs,
            prefill_episodes = args.prefill_episodes,
            seed             = args.seed,
        )