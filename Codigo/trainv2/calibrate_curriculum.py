import numpy as np
from tclab_env import TCLabEnv # Asegúrate de que tclab_env.py tiene los nuevos Uh y K

class PIController:
    """
    Controlador PI discreto sintonizado via reglas IMC de Skogestad
    para el FOPDT: Kp=0.6052, tau=179.08, theta=22.19
    """
    def __init__(self, Kc=7.3, tauI=160.3, dt=1.0):
        self.Kc = Kc
        self.tauI = tauI
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute_action(self, T_ref, T_obs):
        error = T_ref - T_obs
        self.integral += error * self.dt
        
        # Acción de control en % (0 a 100)
        Qu_percent = self.Kc * (error + (1.0 / self.tauI) * self.integral)
        Qu_percent = np.clip(Qu_percent, 0.0, 100.0)
        
        # Mapear de [0, 100]% al action_space [-1, 1] que espera Gym
        action_gym = (Qu_percent / 50.0) - 1.0
        return np.array([action_gym], dtype=np.float32)

def run_calibration(n_episodes=20):
    print("=" * 60)
    print("  Calibración de Umbrales del Curriculum Learning (SAC)")
    print("  Basado en desempeño de PI sintonizado (IMC Skogestad)")
    print("=" * 60)

    # Diccionario para almacenar los nuevos umbrales propuestos
    new_thresholds = {}

    for stage in [1, 2, 3, 4]:
        # seed fijo para reproducibilidad en la evaluación
        env = TCLabEnv(stage=stage, seed=42) 
        pi_ctrl = PIController()
        
        stage_rewards = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            pi_ctrl.reset()
            
            ep_reward_raw = 0.0
            done = False
            
            while not done:
                # El entorno devuelve T_C_obs (normalizado) y T_ref (normalizado)
                # Para el PI, necesitamos desnormalizar u obtener los valores del info
                # En tclab_env.py, info contiene 'T_C_obs' y 'T_ref_C'
                
                # Ejecutamos una acción "dummy" inicial (cero) solo para obtener el info dict completo
                # Alternativamente, reconstruimos desde la observación:
                # obs[3] = (T_C_obs - T_mid) / T_half
                T_mid = (24.5 + 50.0) / 2.0
                T_half = (50.0 - 24.5) / 2.0
                
                T_C_obs = (obs[3] * T_half) + T_mid
                T_ref_C = (obs[4] * T_half) + T_mid
                
                action = pi_ctrl.compute_action(T_ref_C, T_C_obs)
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                # La función compute_reward en tclab_env.py devuelve el raw_reward
                ep_reward_raw += reward 
                done = terminated or truncated
                
            stage_rewards.append(ep_reward_raw)

        mean_reward = np.mean(stage_rewards)
        std_reward = np.std(stage_rewards)
        
        # El umbral para SAC debería ser el ~85% del rendimiento óptimo del PI 
        # para permitir el avance, ya que RL siempre tiene algo de varianza.
        # Como los rewards son negativos, multiplicar por 1.15 hace el requisito un poco más laxo.
        proposed_threshold = mean_reward * 1.15 
        
        new_thresholds[stage] = proposed_threshold

        print(f"Etapa {stage}:")
        print(f"  Media Reward PI : {mean_reward:.1f} ± {std_reward:.1f}")
        print(f"  Umbral Sugerido para SAC: {proposed_threshold:.1f}")
        print("-" * 60)
        
    print("\nReemplaza el diccionario THRESHOLDS_RAW en tclab_env.py con:")
    print("THRESHOLDS_RAW = {")
    print(f"    1: {new_thresholds[1]:.1f},")
    print(f"    2: {new_thresholds[2]:.1f},")
    print(f"    3: {new_thresholds[3]:.1f},")
    print(f"    # Etapa 4 no necesita umbral de salida")
    print("}")

if __name__ == "__main__":
    run_calibration()