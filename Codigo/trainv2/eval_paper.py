import os
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from tclab_env import TCLabEnv, NOMINAL_PARAMS

class PaperEvalEnv(TCLabEnv):
    """
    Entorno modificado estrictamente para evaluación formal.
    Incluye la dimensión 46 (Temperatura Ambiente Ta).
    """
    def __init__(self, Ta_eval=33.0):
        # Ta_eval=33.0 para simular tus condiciones reales exactas
        super().__init__(stage=4, noise_std=0.0, preview_len=15, history_len=25)
        self._dead_steps = 22 
        self.Ta_C = Ta_eval 
        
    def force_reset_with_trajectory(self, T0_C: float, Qu0: float, trajectory: np.ndarray):
        super().reset()
        self.T_ref_trajectory = trajectory
        self.T_ref_C = self.T_ref_trajectory[0]
        self.T_K = T0_C + 273.15
        self.Qu_prev = Qu0
        self._T_buffer = [T0_C] * (self._dead_steps + 1)
        self._action_history = [(Qu0 / 50.0) - 1.0] * self.history_len
        self.eT_integral = 0.0
        self.step_count = 0
        NOMINAL_PARAMS["Ta"] = self.Ta_C + 273.15
        return self._get_obs_with(T0_C, self.T_ref_C - T0_C, 0.0)

    def apply_disturbance(self, dist_C: float):
        NOMINAL_PARAMS["Ta"] += dist_C

def run_test_sequence(model, env, test_name: str, duration: int, steps_logic: callable):
    obs = steps_logic(env, init=True)
    ise = itse = iae = itae = 0.0
    t = 0.0

    for step in range(duration):
        steps_logic(env, t=t)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)
        
        T_C_real = env.T_K - 273.15
        eT = env.T_ref_C - T_C_real
        
        ise += (eT**2)
        itse += t * (eT**2)
        iae += abs(eT)
        itae += t * abs(eT)
        t += env.dt

    return {
        "Test": test_name,
        "ISE": round(ise, 2),
        "ITSE": round(itse, 2),
        "IAE": round(iae, 2),
        "ITAE": round(itae, 2)
    }

# Lógicas adaptadas para T_a = 33°C (T_ref debe estar por encima de Ta)
def logic_test1(env, init=False, t=None):
    if init: 
        # Test 1 adaptado: 45C baja a 40C
        traj = np.ones(300 + 50) * 45.0
        traj[100:] = 40.0
        return env.force_reset_with_trajectory(T0_C=45.0, Qu0=30.0, trajectory=traj)
    if t == 200.0: env.apply_disturbance(-4.0)

def logic_test2(env, init=False, t=None):
    if init: 
        # Test 2 adaptado (como en el hardware real)
        traj = np.ones(500 + 50) * 40.0
        traj[100:200] = 42.0
        traj[200:300] = 45.0
        traj[300:400] = 48.0
        traj[400:] = 40.0
        return env.force_reset_with_trajectory(T0_C=40.0, Qu0=15.0, trajectory=traj)

def logic_test3(env, init=False, t=None):
    if init: 
        # Test 3 adaptado
        traj = np.ones(400 + 50) * 45.0
        return env.force_reset_with_trajectory(T0_C=45.0, Qu0=30.0, trajectory=traj)
    if t == 100.0: env.apply_disturbance(3.0)
    if t == 200.0: env.apply_disturbance(1.0)
    if t == 300.0: env.apply_disturbance(-5.0)

def evaluate_formal_metrics(model_path: str):
    print("=" * 70)
    print("  Evaluación Formal MDPI Processes — Métricas Integrales")
    print("  (Agente SAC Predictivo Híbrido - Ta=33°C)")
    print("=" * 70)
    
    env = PaperEvalEnv(Ta_eval=33.0)
    # Importante: Buscamos el mejor modelo guardado
    model = SAC.load(model_path, env=env, device="cpu")
    
    results = [
        run_test_sequence(model, env, "Test 1: Preliminar (300s)", 300, logic_test1),
        run_test_sequence(model, env, "Test 2: Seguimiento (500s)", 500, logic_test2),
        run_test_sequence(model, env, "Test 3: Disturbios (400s)", 400, logic_test3)
    ]
    
    df = pd.DataFrame(results).set_index("Test")
    print("\nResultados del modelo SAC definitivo:")
    print("-" * 70)
    print(df.to_string())
    print("=" * 70)

if __name__ == "__main__":
    MODEL_PATH = "./models/sac_tclab_final_model.zip" # Revisa en tu carpeta logs/models el nombre exacto
    evaluate_formal_metrics(MODEL_PATH)