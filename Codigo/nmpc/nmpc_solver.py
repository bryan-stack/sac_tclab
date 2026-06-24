"""Solver NMPC para la planta TCLab.

En cada paso resuelve un problema de optimización en horizonte deslizante
sobre el modelo no lineal de la planta. Los parámetros físicos (Uh, K) se
pueden pasar al solver para que use los mismos valores que la planta cuando
el entorno aplica Domain Randomization; el warm start parte de la acción del
NMPC del paso anterior. El horizonte (Tp, Tc) y el paso dt son configurables.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Optional, Dict


class NMPCSolver:
    """
    NMPC para TCLab. Resuelve cada llamada:
      min  Σ S·(T_ref - T)² + R·Qu²  +  P_N·(T_ref - T_Tp)²
      Qu
      s.t. dT/dt = f(T, Qu)         modelo no-lineal
           Qu ∈ [0, 100]            límites de actuador

    Horizontes:
      Tp = 20s (predicción)
      Tc = 5s  (control)
    Pesos (Tabla 3 del paper):
      S = 0.0113, R = 0.001, P_N = 0.1513
    """

    def __init__(
        self,
        dt:        float = 1.0,
        Tp:        int   = 20,
        Tc:        int   = 5,
        S:         float = 0.0113,
        R:         float = 0.001,
        P_N:       float = 0.1513,
        params:    Optional[Dict[str, float]] = None,
    ):
        self.dt  = dt
        self.Tp  = Tp
        self.Tc  = Tc
        self.S   = S
        self.R   = R
        self.P_N = P_N

        # parámetros físicos inyectables. Si no se pasan, se usan los nominales.
        # Durante el entrenamiento, el entorno re-instanciará el solver o
        # actualizará los parámetros en cada reset() para reflejar el DR.
        self.params = params if params is not None else {
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

    def update_params(self, **new_params) -> None:
        """Actualiza los parámetros del modelo (llamado por el entorno al hacer DR)."""
        self.params.update(new_params)

    def _thermal_dynamics(self, T_K: float, Qu: float) -> float:
        p = self.params
        Q_conv = p["Uh"] * p["A"] * (p["Ta"] - T_K)
        Q_rad  = p["eps"] * p["A"] * p["sigma"] * (p["Ta"]**4 - T_K**4)
        Q_heat = p["alpha"] * p["K"] * Qu
        return (Q_conv + Q_rad + Q_heat) / (p["m"] * p["cp"])

    def _cost_function(self, U: np.ndarray, T0_K: float, T_ref_K: float) -> float:
        """Costo del NMPC: J = Σ [S·(T_ref - T_k)² + R·Qu_norm²] + P_N·(T_ref - T_Tp)².

        Qu se normaliza a [0,1] antes de evaluar el costo de control. Con Qu en
        [0,100] el término R·Qu² dominaría sobre S·eT² y el optimizador tendería
        a Qu≈0, dejando al NMPC sin capacidad de calentar.
        """
        J   = 0.0
        T_k = T0_K
        for k in range(self.Tp):
            Qu = U[k] if k < self.Tc else U[-1]
            Qu_norm = Qu / 100.0
            dTdt = self._thermal_dynamics(T_k, Qu)
            T_k += dTdt * self.dt
            error = T_ref_K - T_k
            J += self.S * (error**2) + self.R * (Qu_norm**2)
        # Costo terminal
        J += self.P_N * (T_ref_K - T_k)**2
        return J

    def compute_action(
        self,
        T0_C:    float,
        T_ref_C: float,
        Q_warm:  float = 30.0,
    ) -> float:
        """
        Resuelve el NMPC y devuelve la primera acción del horizonte.

        Args:
            T0_C:    temperatura actual (°C)
            T_ref_C: setpoint (°C)
            Q_warm:  acción del NMPC anterior, para warm start

        Returns:
            Q_NMPC ∈ [0, 100]
        """
        T0_K     = T0_C + 273.15
        T_ref_K  = T_ref_C + 273.15

        U0 = np.full(self.Tc, Q_warm)
        bounds = [(0.0, 100.0) for _ in range(self.Tc)]

        res = minimize(
            fun     = self._cost_function,
            x0      = U0,
            args    = (T0_K, T_ref_K),
            bounds  = bounds,
            method  = 'SLSQP',
            options = {'ftol': 1e-3, 'maxiter': 30, 'disp': False},
        )

        return float(res.x[0]) if res.success else float(Q_warm)