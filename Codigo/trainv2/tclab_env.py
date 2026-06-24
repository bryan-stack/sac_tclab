"""Entorno Gymnasium de la planta TCLab para entrenar el agente SAC.

La planta se simula con la ODE no lineal de balance de energía, con tiempo
muerto, ruido de medición y Domain Randomization. La observación tiene 46
dimensiones (5 estados de error, la temperatura ambiente, 25 acciones previas
y 15 pasos de previsualización de la referencia) y la recompensa penaliza de
forma asimétrica el sobreimpulso. La dificultad se organiza en cuatro etapas
de Curriculum Learning, con umbrales de avance fijados respecto al PI.
"""

import numpy as np
from typing import Optional, Tuple, Dict
import gymnasium as gym
from gymnasium import spaces

# 1. PARÁMETROS FÍSICOS (calibrados por System Identification)

NOMINAL_PARAMS: Dict[str, float] = {
    "Ta":    273.15 + 26.75,   # K — temperatura ambiente nominal
    "alpha": 0.019,            # W/(% heater)
    "cp":    10_000.0,         # J/(kg·K)
    "A":     0.012,            # m²
    "m":     0.004,            # kg
    "Uh":    11.8628,          # W/(m²·K) — calibrado: Kp=0.6052, τ=179.08s
    "eps":   0.9,
    "sigma": 5.67e-8,          # W/(m²·K⁴)
    "K":     6.9655,           # factor de corrección de potencia
}

T_MIN_C: float = 24.5
T_MAX_C: float = 50.0
QU_MIN:  float = 0.0
QU_MAX:  float = 100.0

# Rango de temperatura ambiente cubierto durante el entrenamiento 
# Colombia (La Paz) puede tener Ta∈[20,35]°C según el clima
TA_MIN_C: float = 20.0
TA_MAX_C: float = 35.0

# 2. MODELO TÉRMICO NO-LINEAL (Ec. 2 del paper de referencia)

def nonlinear_thermal_ode(
    T_K:    float,
    Qu:     float,
    params: Dict[str, float],
    T_dist: float = 0.0,
) -> float:
    """
    dT/dt = [Uh·A·(Ta-T) + ε·A·σ·(Ta⁴-T⁴) + α·K·Qu] / (m·cp)

    T_dist: perturbación aditiva sobre Ta (°C o K, mismas unidades).
    Devuelve dT/dt en K/s.
    """
    Ta      = params["Ta"] + T_dist
    Q_conv  = params["Uh"] * params["A"] * (Ta - T_K)
    Q_rad   = params["eps"] * params["A"] * params["sigma"] * (Ta**4 - T_K**4)
    Q_heat  = params["alpha"] * params["K"] * Qu
    return (Q_conv + Q_rad + Q_heat) / (params["m"] * params["cp"])


# 3. ENTORNO GYMNASIUM

class TCLabEnv(gym.Env):
    """
    Entorno TCLab v4 para entrenamiento con SAC + Curriculum Learning.

    Espacio de observación (46 dimensiones):
        [0]    eT          — error de temperatura normalizado [-1,1]
        [1]    ∫eT·dt      — error integral normalizado
        [2]    deT/dt      — error derivativo normalizado
        [3]    T_obs       — temperatura observada (retardada) normalizada
        [4]    T_ref       — setpoint actual normalizado
        [5]    Ta_norm     — temperatura ambiente normalizada [M4]
        [6:31] hist_Qu     — historial de 25 acciones previas (propiedad Markov)
        [31:46] preview    — previsualización de 15 pasos de T_ref (lookahead)

    Espacio de acción (1 dimensión):
        a ∈ [-1, 1]  →  Qu = 50·(a+1) ∈ [0, 100] %
    """

    metadata = {"render_modes": []}

    # ── Umbrales recalibrados para:                                       ──
    #    • episode_len=500 en stages 2-4                                   ──
    #    • T0 distribuido parcialmente desde Ta (cold-start)               ──
    #    • Ta aleatorio en [20,35]°C                                       ──
    #    Valor = 115% de la media del controlador PI (Skogestad IMC),      ──
    #    es decir: "el SAC debe ser al menos tan bueno como el 87% del PI" ──
    THRESHOLDS_RAW = {
        1: -166.0,   # Igual al v3 que produjo ISE=40.85 — NO cambiar
        2: -200.0,   # Stage 2 con 500s es más largo → reward más negativo
        3: -340.0,   # Stage 3 relajado (agente llega a -326 pero con
                     # ent_coef descontrolado; con ceiling debería mejorar)
    }

    # ── Configuración por etapa ──────────────────────────────────────────
    # Stage 1 IDÉNTICO al v3 original. Mejoras M2-M5 solo en stages 2+.
    STAGE_CONFIG = {
        1: {
            # IDÉNTICO al v3 que produjo Test3-ISE=40.85
            "T_ref_range":     (40.0, 40.0),
            "dist_max":        0.0,
            "dead_time_range": (0, 0),
            "DR":              0.00,
            "episode_len":     300,
            "T0_std":          2.0,          # original — NO cambiar
            "cold_start_prob": 0.0,          # sin cold-start en Stage 1
        },
        2: {
            "T_ref_range":     (35.0, 45.0),
            "dist_max":        1.0,
            "dead_time_range": (11, 11),
            "DR":              0.05,
            "episode_len":     500,          # [M2]
            "T0_std":          3.0,
            "cold_start_prob": 0.2,          # [M3] leve
        },
        3: {
            "T_ref_range":     (30.0, 48.0),
            "dist_max":        3.0,
            "dead_time_range": (0, 22),
            "DR":              0.10,
            "episode_len":     500,          # [M2]
            "T0_std":          4.0,
            "cold_start_prob": 0.4,          # [M3] moderado
        },
        4: {
            "T_ref_range":     (30.0, 48.0),
            "dist_max":        5.0,
            "dead_time_range": (0, 33),
            "DR":              0.15,
            "episode_len":     500,          # [M2]
            "T0_std":          5.0,
            "cold_start_prob": 0.5,          # [M3] máximo
        },
    }

    HISTORY_LEN  = 25   # pasos de historial de acciones
    PREVIEW_LEN  = 15   # pasos de lookahead de referencia
    OBS_SIZE     = 5 + 1 + HISTORY_LEN + PREVIEW_LEN  # 46 = 5+1+25+15 [M4]

    def __init__(
        self,
        stage:       int   = 1,
        dt:          float = 1.0,
        noise_std:   float = 0.10,
        seed:        Optional[int] = None,
    ):
        super().__init__()
        assert stage in (1, 2, 3, 4), "stage debe ser 1, 2, 3 o 4"
        self.stage     = stage
        self.dt        = dt
        self.noise_std = noise_std

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Constantes de normalización
        self._T_mid  = (T_MIN_C + T_MAX_C) / 2.0        # 37.25°C
        self._T_half = (T_MAX_C - T_MIN_C) / 2.0        # 12.75°C
        self._Ta_mid  = (TA_MIN_C + TA_MAX_C) / 2.0     # 27.5°C [M4]
        self._Ta_half = (TA_MAX_C - TA_MIN_C) / 2.0     # 7.5°C  [M4]

        # Variables de estado (se inicializan en reset)
        self.episode_len:      int   = self.STAGE_CONFIG[stage]["episode_len"]
        self.T_K:              float = 0.0
        self.T_ref_C:          float = 40.0
        self.Ta_C:             float = 26.75   # temperatura ambiente del episodio [M4]
        self.eT_integral:      float = 0.0
        self.eT_prev:          float = 0.0
        self.Qu_prev:          float = 0.0
        self.step_count:       int   = 0
        self._dead_steps:      int   = 0
        self._T_buffer:        list  = []
        self._action_history:  list  = []
        self.T_ref_trajectory: np.ndarray = np.array([])

        if seed is not None:
            np.random.seed(seed)

    # ── Reset ────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        cfg = self.STAGE_CONFIG[self.stage]
        self.episode_len = cfg["episode_len"]

        # Sortear temperatura ambiente para este episodio
        self.Ta_C = float(np.random.uniform(TA_MIN_C, TA_MAX_C))
        # Actualizar parámetros del simulador con la nueva Ta
        self._params = dict(NOMINAL_PARAMS)
        self._params["Ta"] = 273.15 + self.Ta_C
        # Aplicar Domain Randomization sobre Uh y K
        dr = cfg["DR"]
        if dr > 0:
            self._params["Uh"] *= np.random.uniform(1 - dr, 1 + dr)
            self._params["K"]  *= np.random.uniform(1 - dr, 1 + dr)

        # Construir trayectoria de referencia para el episodio completo + preview
        total_len = self.episode_len + self.PREVIEW_LEN + 1
        base_ref  = float(np.random.uniform(*cfg["T_ref_range"]))
        self.T_ref_trajectory = np.full(total_len, base_ref)

        # En stages 2+ introducir escalones con probabilidad 0.7 
        if self.stage >= 2 and np.random.rand() > 0.3:
            # Hasta 3 escalones aleatorios en el episodio
            n_steps = np.random.randint(1, 4)
            step_times = sorted(np.random.randint(50, self.episode_len - 50, size=n_steps))
            for t in step_times:
                new_ref = float(np.random.uniform(*cfg["T_ref_range"]))
                self.T_ref_trajectory[t:] = new_ref

        self.T_ref_C = self.T_ref_trajectory[0]

        # Condición inicial: cold-start (desde Ta) o normal (desde T_ref ± std)
        if np.random.rand() < cfg["cold_start_prob"]:
            # Cold-start: simular una placa que está en equilibrio térmico con el ambiente
            T0_C = float(np.clip(self.Ta_C + np.random.uniform(0, 4), T_MIN_C, T_MAX_C))
        else:
            T0_C = float(np.clip(
                self.T_ref_C + np.random.normal(0, cfg["T0_std"]),
                T_MIN_C, T_MAX_C
            ))

        self.T_K         = T0_C + 273.15
        self.eT_integral = 0.0
        self.eT_prev     = self.T_ref_C - T0_C
        self.Qu_prev     = 30.0
        self.step_count  = 0

        # Dead time buffer
        dt_lo, dt_hi = cfg["dead_time_range"]
        self._dead_steps = int(np.random.randint(dt_lo, dt_hi + 1)) if dt_hi > dt_lo else dt_lo
        self._T_buffer   = [T0_C] * (self._dead_steps + 1)

        # Historial de acciones (inicializado en Qu=30% normalizado = -0.4)
        init_action_norm = (30.0 / 50.0) - 1.0   # ≈ -0.4
        self._action_history = [init_action_norm] * self.HISTORY_LEN

        return self._get_obs(), {}

    # ── Step ─────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # 1. Setpoint actual desde la trayectoria pre-generada
        idx = min(self.step_count, len(self.T_ref_trajectory) - 1)
        self.T_ref_C = float(self.T_ref_trajectory[idx])

        # 2. Acción → Qu
        Qu = float(np.clip((action[0] + 1.0) * 50.0, QU_MIN, QU_MAX))

        # 3. Disturbio estocástico sobre Ta
        cfg   = self.STAGE_CONFIG[self.stage]
        T_dist = float(np.random.uniform(-cfg["dist_max"], cfg["dist_max"])) \
                 if cfg["dist_max"] > 0 else 0.0

        # 4. Integración del ODE (Euler explícito)
        dTdt    = nonlinear_thermal_ode(self.T_K, Qu, self._params, T_dist)
        self.T_K = float(self.T_K + dTdt * self.dt)
        T_C_real = self.T_K - 273.15 + np.random.normal(0, self.noise_std)

        # 5. Dead-time buffer
        self._T_buffer.append(T_C_real)
        T_C_obs = self._T_buffer.pop(0) if self._dead_steps > 0 else T_C_real

        # 6. Historial de acciones
        self._action_history.append(float(action[0]))
        self._action_history.pop(0)

        # 7. Errores
        eT_obs  = self.T_ref_C - T_C_obs
        eT_real = self.T_ref_C - T_C_real
        eT_der  = (eT_obs - self.eT_prev) / self.dt
        self.eT_integral += eT_obs * self.dt
        self.eT_prev = eT_obs

        # 8. Recompensa con penalización anti-overshoot 
        r_nmpc    = -(0.01 * eT_real**2 + 0.0001 * Qu**2)
        r_shaped  = -0.3 * (1.0 * abs(eT_obs) + 0.1 * abs(Qu - self.Qu_prev))
        # Penalizar superar el setpoint asimétrica y fuertemente.
        #   Cuando T > T_ref (overshoot), el agente recibe penalización extra.
        #   Coeficiente 0.02: mayor que S=0.01 para disuadir el overshoot,
        #   menor que 0.05 para no dominar sobre el objetivo de tracking.
        r_anti_os = -0.02 * max(0.0, T_C_real - self.T_ref_C)**2

        reward = r_nmpc + r_shaped + r_anti_os

        self.Qu_prev = Qu
        self.step_count += 1

        terminated = False
        truncated  = self.step_count >= self.episode_len

        info = {
            "eT":       eT_real,
            "eT_obs":   eT_obs,
            "Qu":       Qu,
            "T_C_real": T_C_real,
            "T_C_obs":  T_C_obs,
            "Ta_C":     self.Ta_C,
        }

        return self._build_obs(T_C_obs, eT_obs, eT_der), reward, terminated, truncated, info

    # ── Construcción de observación ──────────────────────────────────────
    def _build_obs(self, T_obs: float, eT: float, eT_der: float) -> np.ndarray:
        """
        46 dimensiones: [eT, ∫eT, deT, T_obs, T_ref, Ta_norm, hist×25, preview×15]
        """
        # 5 estados de error estándar
        base = [
            float(np.clip(eT                 / 50.0,    -1, 1)),
            float(np.clip(self.eT_integral   / 15000.0, -1, 1)),
            float(np.clip(eT_der             / 2.0,     -1, 1)),
            float(np.clip((T_obs - self._T_mid)   / self._T_half,  -1, 1)),
            float(np.clip((self.T_ref_C - self._T_mid) / self._T_half, -1, 1)),
        ]

        # Temperatura ambiente normalizada (1 dimensión)
        ta_norm = float(np.clip((self.Ta_C - self._Ta_mid) / self._Ta_half, -1, 1))

        # Historial de acciones (25 dimensiones)
        hist = list(self._action_history)

        # Lookahead de referencia (15 dimensiones)
        preview = []
        for i in range(self.PREVIEW_LEN):
            idx = min(self.step_count + i, len(self.T_ref_trajectory) - 1)
            v   = float(np.clip(
                (self.T_ref_trajectory[idx] - self._T_mid) / self._T_half, -1, 1
            ))
            preview.append(v)

        return np.array(base + [ta_norm] + hist + preview, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Observación inicial en reset (sin pasos previos)."""
        T_init = self._T_buffer[0] if self._T_buffer else self.T_K - 273.15
        eT_obs = self.T_ref_C - T_init
        return self._build_obs(T_init, eT_obs, 0.0)

    def set_stage(self, stage: int) -> None:
        assert stage in (1, 2, 3, 4)
        self.stage      = stage
        self.episode_len = self.STAGE_CONFIG[stage]["episode_len"]

    def render(self) -> None:
        pass


# 4. CALLBACK DE CURRICULUM (versión compacta para importar desde train_sac_v3)

try:
    from stable_baselines3.common.callbacks import BaseCallback

    class CurriculumCallback(BaseCallback):
        """Versión ligera para uso autónomo. Ver SyncedCurriculumCallback en train_sac_v3."""
        def __init__(self, verbose: int = 1):
            super().__init__(verbose)
            self.window        = 3
            self._eval_rewards = []

        def _on_step(self) -> bool:
            eval_cb = self.locals.get("callback")
            if hasattr(eval_cb, "last_mean_reward"):
                res = eval_cb.last_mean_reward
                if res > -10000.0 and (not self._eval_rewards or res != self._eval_rewards[-1]):
                    self._eval_rewards.append(res)
                    if len(self._eval_rewards) > self.window:
                        self._eval_rewards.pop(0)
                    stage = self.training_env.get_attr("stage")[0]
                    if stage < 4 and len(self._eval_rewards) == self.window:
                        mean_r = np.mean(self._eval_rewards)
                        if mean_r >= TCLabEnv.THRESHOLDS_RAW[stage]:
                            new_stage = stage + 1
                            self.training_env.env_method("set_stage", new_stage)
                            self._eval_rewards.clear()
                            if self.verbose:
                                print(f"¡AVANCE A ETAPA {new_stage}! (media={mean_r:.1f})")
            return True

except ImportError:
    pass