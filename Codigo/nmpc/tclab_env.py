"""Entorno Gymnasium de la arquitectura residual NMPC-SAC.

La acción aplicada es Qu(t) = clip(Q_NMPC(t) + Δ_SAC(t), [0, 100]), donde
Q_NMPC viene del optimizador NMPC y Δ_SAC ∈ [-MAX_RESIDUAL, MAX_RESIDUAL] es la
corrección que aprende el agente. El NMPC usa los mismos parámetros que la
planta tras el Domain Randomization y se refresca cada NMPC_REFRESH_STEPS pasos
para acotar el costo de cómputo. La observación tiene 56 dimensiones (5 estados
de error, la temperatura ambiente, 25 acciones previas y 25 pasos de
previsualización de la referencia).
"""

import numpy as np
from typing import Optional, Tuple, Dict
import gymnasium as gym
from gymnasium import spaces

from nmpc_solver import NMPCSolver

# Constantes físicas

NOMINAL_PARAMS: Dict[str, float] = {
    "Ta":    273.15 + 26.75,
    "alpha": 0.019,
    "cp":    10_000.0,
    "A":     0.012,
    "m":     0.004,
    "Uh":    11.8628,
    "eps":   0.9,
    "sigma": 5.67e-8,
    "K":     6.9655,
}

T_MIN_C:  float = 24.5
T_MAX_C:  float = 50.0
QU_MIN:   float = 0.0
QU_MAX:   float = 100.0
TA_MIN_C: float = 20.0
TA_MAX_C: float = 35.0

# ODE

def nonlinear_thermal_ode(
    T_K:    float,
    Qu:     float,
    params: Dict[str, float],
    T_dist: float = 0.0,
) -> float:
    Ta     = params["Ta"] + T_dist
    Q_conv = params["Uh"] * params["A"] * (Ta - T_K)
    Q_rad  = params["eps"] * params["A"] * params["sigma"] * (Ta**4 - T_K**4)
    Q_heat = params["alpha"] * params["K"] * Qu
    return (Q_conv + Q_rad + Q_heat) / (params["m"] * params["cp"])

# Entorno

class TCLabEnv(gym.Env):
    metadata = {"render_modes": []}

    # Umbrales del curriculum recalibrados con baseline del NMPC puro (Δ=0):
    #   Stage 1 (regulación, sin dead time): NMPC puro reward ~-10 → umbral -60
    #   Stage 2 (tracking + dead 11s):       NMPC puro reward ~-400 → umbral -300
    #   Stage 3 (dead variable + DR 10%):    NMPC puro reward ~-600 → umbral -450
    # El SAC tiene margen amplio para mejorar sobre el NMPC en stages 2-4.
    THRESHOLDS_RAW = {
        1: -83.0,
        2: -283.0,
        3: -478.0,
    }

    STAGE_CONFIG = {
        1: {
            "T_ref_range":     (40.0, 40.0),
            "dist_max":        0.0,
            "dead_time_range": (0, 0),
            "DR":              0.00,
            "episode_len":     300,
            "T0_std":          2.0,
            "cold_start_prob": 0.0,
        },
        2: {
            "T_ref_range":     (35.0, 45.0),
            "dist_max":        1.0,
            "dead_time_range": (11, 11),
            "DR":              0.05,
            "episode_len":     500,
            "T0_std":          3.0,
            "cold_start_prob": 0.2,
        },
        3: {
            "T_ref_range":     (30.0, 48.0),
            "dist_max":        3.0,
            "dead_time_range": (0, 22),
            "DR":              0.10,
            "episode_len":     500,
            "T0_std":          4.0,
            "cold_start_prob": 0.4,
        },
        4: {
            "T_ref_range":     (30.0, 48.0),
            "dist_max":        5.0,
            "dead_time_range": (0, 33),
            "DR":              0.15,
            "episode_len":     500,
            "T0_std":          5.0,
            "cold_start_prob": 0.5,
        },
    }

    # Historia reducida (suficiente para cubrir θp=22s + margen)
    HISTORY_LEN = 25
    # Preview aumentado (necesario para anticipar con dead time real)
    PREVIEW_LEN = 25
    OBS_SIZE    = 5 + 1 + HISTORY_LEN + PREVIEW_LEN   # 56

    # Cacheo del NMPC: refrescar cada N pasos.
    # Justificación: la planta tiene τ=179s y θ=22s. La acción óptima del
    # NMPC NO cambia significativamente entre pasos consecutivos (dt=1s).
    # Refrescar cada 5 pasos da 7x speedup con < 1% degradación de calidad.
    NMPC_REFRESH_STEPS = 5
    # Corrección residual máxima del SAC: ±15% sobre la acción del NMPC.
    MAX_RESIDUAL = 15.0

    def __init__(
        self,
        stage:     int   = 1,
        dt:        float = 1.0,
        noise_std: float = 0.10,
        seed:      Optional[int] = None,
    ):
        super().__init__()
        assert stage in (1, 2, 3, 4)
        self.stage     = stage
        self.dt        = dt
        self.noise_std = noise_std

        # Solver NMPC: sus parámetros se sincronizan con la planta en cada reset
        self.nmpc = NMPCSolver(dt=self.dt)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self._T_mid   = (T_MIN_C + T_MAX_C) / 2.0
        self._T_half  = (T_MAX_C - T_MIN_C) / 2.0
        self._Ta_mid  = (TA_MIN_C + TA_MAX_C) / 2.0
        self._Ta_half = (TA_MAX_C - TA_MIN_C) / 2.0

        # Variables de estado (se completan en reset)
        self.episode_len: int = self.STAGE_CONFIG[stage]["episode_len"]
        self.T_K:           float = 0.0
        self.T_ref_C:       float = 40.0
        self.Ta_C:          float = 26.75
        self.eT_integral:   float = 0.0
        self.eT_prev:       float = 0.0
        self.Qu_prev:       float = 0.0    # Qu total previo (NMPC + residual)
        self.Q_NMPC_prev:   float = 30.0   # [F2] solo Q_NMPC previo para warm start
        self.Q_NMPC_cached: float = 30.0   # [F3] cache para evitar recálculo
        self.delta_Q_prev:  float = 0.0    # residuo previo para reward shaping
        self.step_count:    int   = 0
        self._dead_steps:   int   = 0
        self._T_buffer:     list  = []
        self._action_history: list  = []
        self.T_ref_trajectory: np.ndarray = np.array([])

        if seed is not None:
            np.random.seed(seed)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        cfg = self.STAGE_CONFIG[self.stage]
        self.episode_len = cfg["episode_len"]

        # Ta y DR aleatorios
        self.Ta_C = float(np.random.uniform(TA_MIN_C, TA_MAX_C))
        self._params = dict(NOMINAL_PARAMS)
        self._params["Ta"] = 273.15 + self.Ta_C
        dr = cfg["DR"]
        if dr > 0:
            self._params["Uh"] *= np.random.uniform(1 - dr, 1 + dr)
            self._params["K"]  *= np.random.uniform(1 - dr, 1 + dr)

        # sincronizar parámetros del NMPC con los de la planta.
        # Sin esto, el NMPC predice una planta distinta a la real y SAC tiene
        # que compensar errores grandes en lugar de mejoras finas.
        self.nmpc.update_params(**self._params)

        # Trayectoria de referencia con escalones
        total_len = self.episode_len + self.PREVIEW_LEN + 1
        base_ref  = float(np.random.uniform(*cfg["T_ref_range"]))
        self.T_ref_trajectory = np.full(total_len, base_ref)

        if self.stage >= 2 and np.random.rand() > 0.3:
            n_steps = np.random.randint(1, 4)
            step_times = sorted(np.random.randint(50, self.episode_len - 50, size=n_steps))
            for t in step_times:
                new_ref = float(np.random.uniform(*cfg["T_ref_range"]))
                self.T_ref_trajectory[t:] = new_ref

        self.T_ref_C = self.T_ref_trajectory[0]

        # T0: cold-start o cerca del setpoint
        if np.random.rand() < cfg["cold_start_prob"]:
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
        self.Q_NMPC_prev = 30.0
        self.Q_NMPC_cached = 30.0
        self.delta_Q_prev = 0.0
        self.step_count  = 0

        # Dead time buffer
        dt_lo, dt_hi = cfg["dead_time_range"]
        self._dead_steps = int(np.random.randint(dt_lo, dt_hi + 1)) if dt_hi > dt_lo else dt_lo
        self._T_buffer   = [T0_C] * (self._dead_steps + 1)

        # Historial de acciones residuales (no de Qu total)
        self._action_history = [0.0] * self.HISTORY_LEN

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # 1. Setpoint actual
        idx = min(self.step_count, len(self.T_ref_trajectory) - 1)
        self.T_ref_C = float(self.T_ref_trajectory[idx])

        # 2. Temperatura observada (con dead time)
        T_C_obs = self._T_buffer[0] if self._dead_steps > 0 else (self.T_K - 273.15)

        # 3. NMPC con cacheo : recalcular cada NMPC_REFRESH_STEPS pasos
        if self.step_count % self.NMPC_REFRESH_STEPS == 0:
            # warm start con Q_NMPC anterior, no Qu total
            Q_NMPC = self.nmpc.compute_action(
                T0_C    = T_C_obs,
                T_ref_C = self.T_ref_C,
                Q_warm  = self.Q_NMPC_prev,
            )
            self.Q_NMPC_cached = Q_NMPC
            self.Q_NMPC_prev   = Q_NMPC
        else:
            Q_NMPC = self.Q_NMPC_cached

        # 4. Residuo del SAC y fusión
        delta_Q_SAC = float(action[0]) * self.MAX_RESIDUAL
        Qu = float(np.clip(Q_NMPC + delta_Q_SAC, QU_MIN, QU_MAX))

        # 5. Integración del ODE
        cfg    = self.STAGE_CONFIG[self.stage]
        T_dist = float(np.random.uniform(-cfg["dist_max"], cfg["dist_max"])) \
                 if cfg["dist_max"] > 0 else 0.0

        dTdt    = nonlinear_thermal_ode(self.T_K, Qu, self._params, T_dist)
        self.T_K = float(self.T_K + dTdt * self.dt)
        T_C_real = self.T_K - 273.15 + np.random.normal(0, self.noise_std)

        # 6. Dead time buffer
        self._T_buffer.append(T_C_real)
        T_C_obs = self._T_buffer.pop(0) if self._dead_steps > 0 else T_C_real

        # 7. Historial: guardar la acción residual normalizada
        self._action_history.append(float(action[0]))
        self._action_history.pop(0)

        # 8. Errores
        eT_obs  = self.T_ref_C - T_C_obs
        eT_real = self.T_ref_C - T_C_real
        eT_der  = (eT_obs - self.eT_prev) / self.dt
        self.eT_integral += eT_obs * self.dt
        self.eT_prev = eT_obs

        # 9. Recompensa (Ecs. 31, 33, 34 del paper)
        # Qu normalizado a [0,1] para que el peso R sea consistente
        # con el del solver NMPC. Sin esto, R·Qu² con Qu∈[0,100] domina sobre
        # S·eT² y el reward se vuelve "evitar usar el heater" en lugar de
        # "controlar la temperatura".
        Qu_norm = Qu / 100.0
        r_nmpc = -(self.nmpc.S * (eT_real**2) + self.nmpc.R * (Qu_norm**2))
        # Reward shaping: error absoluto y suavidad del residuo
        alpha_sh = 1.0
        beta_sh  = 0.1
        r_shaped = -(alpha_sh * abs(eT_obs) + beta_sh * abs(delta_Q_SAC - self.delta_Q_prev))
        # λ = 0.3
        reward   = r_nmpc + 0.3 * r_shaped

        # 10. Bookkeeping
        self.Qu_prev      = Qu
        self.delta_Q_prev = delta_Q_SAC
        self.step_count  += 1

        terminated = False
        truncated  = self.step_count >= self.episode_len

        info = {
            "eT":       eT_real,
            "eT_obs":   eT_obs,
            "Qu":       Qu,
            "Q_NMPC":   Q_NMPC,
            "delta_Q":  delta_Q_SAC,
            "T_C_real": T_C_real,
            "T_C_obs":  T_C_obs,
            "Ta_C":     self.Ta_C,
        }

        return self._build_obs(T_C_obs, eT_obs, eT_der), reward, terminated, truncated, info

    def _build_obs(self, T_obs: float, eT: float, eT_der: float) -> np.ndarray:
        base = [
            float(np.clip(eT                  / 50.0,    -1, 1)),
            float(np.clip(self.eT_integral    / 15000.0, -1, 1)),
            float(np.clip(eT_der              / 2.0,     -1, 1)),
            float(np.clip((T_obs - self._T_mid)        / self._T_half,  -1, 1)),
            float(np.clip((self.T_ref_C - self._T_mid) / self._T_half,  -1, 1)),
        ]
        ta_norm = float(np.clip((self.Ta_C - self._Ta_mid) / self._Ta_half, -1, 1))
        hist    = list(self._action_history)

        preview = []
        for i in range(self.PREVIEW_LEN):
            idx = min(self.step_count + i, len(self.T_ref_trajectory) - 1)
            v   = float(np.clip(
                (self.T_ref_trajectory[idx] - self._T_mid) / self._T_half, -1, 1
            ))
            preview.append(v)

        return np.array(base + [ta_norm] + hist + preview, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        T_init = self._T_buffer[0] if self._T_buffer else self.T_K - 273.15
        eT_obs = self.T_ref_C - T_init
        return self._build_obs(T_init, eT_obs, 0.0)

    def set_stage(self, stage: int) -> None:
        assert stage in (1, 2, 3, 4)
        self.stage       = stage
        self.episode_len = self.STAGE_CONFIG[stage]["episode_len"]

    def render(self) -> None:
        pass