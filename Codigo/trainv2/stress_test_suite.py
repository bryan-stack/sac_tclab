"""
stress_test_suite.py — Batalla Final en Hardware (PI vs SAC Puro)
=================================================================
Ejecuta el perfil dinámico de estrés térmico para comparar
el control clásico contra el agente SAC Model-Free.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import tclab
from tclab import clock

from tclab_env import TCLabEnv

# 1. PERFIL DINÁMICO DE ESTRÉS
def obtener_referencia_dinamica(t):
    if t < 150:
        return 35.0
    elif t < 300:
        return 48.0
    elif t < 450:
        return 28.0
    else:
        return 40.0

# 2. CONTROLADORES
class PIController:
    def __init__(self, Kc=7.3, tauI=160.3, dt=1.0):
        self.Kc = Kc
        self.tauI = tauI
        self.dt = dt
        self._integral = 0.0

    def compute(self, T_ref, T_obs):
        e = T_ref - T_obs
        self._integral += e * self.dt
        Qu = self.Kc * (e + self._integral / self.tauI)
        
        # Anti-windup
        if Qu > 100.0:
            Qu = 100.0
            self._integral -= e * self.dt 
        elif Qu < 0.0:
            Qu = 0.0
            self._integral -= e * self.dt
        return Qu

class SACHardwareAdapter(TCLabEnv):
    """Adaptador para el agente SAC Puro (Model-Free)"""
    def __init__(self, stage=4):
        super().__init__(stage=stage)

    def reset_hardware(self, real_T0):
        super().reset()
        self.T_K = real_T0 + 273.15
        self.Ta_C = real_T0
        self.eT_prev = 0.0
        self.eT_integral = 0.0
        # El SAC Puro asume inicio en 0 (normalizado)
        self._action_history = [0.0] * self.HISTORY_LEN
        self.step_count = 0
        return self._build_obs(real_T0, self.eT_prev, 0.0)

    def step_hardware(self, action, real_T, T_ref_dinamico):
        self.T_ref_C = T_ref_dinamico
        
        # El SAC Puro emite [-1, 1]. Lo mapeamos a [0, 100]
        Qu = float(np.clip((action[0] + 1.0) * 50.0, 0.0, 100.0))

        self._action_history.append(float(action[0]))
        self._action_history.pop(0)

        eT_real = self.T_ref_C - real_T
        eT_der  = (eT_real - self.eT_prev) / self.dt
        self.eT_integral += eT_real * self.dt
        self.eT_prev = eT_real

        self.step_count += 1
        obs = self._build_obs(real_T, eT_real, eT_der)
        return obs, Qu

# 3. MOTOR DE EJECUCIÓN FÍSICA
def ejecutar_prueba(modo, model_path=None, duration=600):
    print(f"\n{'='*50}")
    print(f" INICIANDO PRUEBA DE ESTRÉS: {modo.upper()}")
    print(f"{'='*50}\n")

    log_time, log_Tref, log_Treal, log_Qu = [], [], [], []

    if modo == "sac":
        model = SAC.load(model_path, device="cpu")
        env = SACHardwareAdapter()
        env.episode_len = duration
    else:
        pi = PIController()

    try:
        with tclab.TCLab() as lab:
            T0 = lab.T1
            print(f"[+] Temperatura inicial: {T0:.2f} °C (Verifica que esté fría ~30°C o menos)")
            
            if modo == "sac":
                obs = env.reset_hardware(real_T0=T0)
            
            for t in clock(duration, 1.0):
                T_real = lab.T1
                T_ref = obtener_referencia_dinamica(t)
                
                if modo == "sac":
                    action, _ = model.predict(obs, deterministic=True)
                    obs, Qu = env.step_hardware(action, T_real, T_ref)
                else:
                    Qu = pi.compute(T_ref, T_real)
                
                lab.Q1(Qu)
                
                log_time.append(t)
                log_Tref.append(T_ref)
                log_Treal.append(T_real)
                log_Qu.append(Qu)
                
                if t % 10 == 0:
                    print(f"t: {t:3.0f}s | Ref: {T_ref:5.1f}°C | Real: {T_real:5.2f}°C | Qu: {Qu:5.1f}%")

    except KeyboardInterrupt:
        print("\n[!] Prueba abortada manualmente.")
    
    df = pd.DataFrame({
        "Time(s)": log_time, "Reference(C)": log_Tref, 
        "Temperature(C)": log_Treal, "Control_Effort(%)": log_Qu
    })
    df.to_csv(f"stress_data_{modo}.csv", index=False)
    print(f"\n[OK] Datos guardados en stress_data_{modo}.csv")

# 4. GRAFICADOR COMPARATIVO
def graficar_comparativa():
    if not os.path.exists("stress_data_pi.csv") or not os.path.exists("stress_data_sac.csv"):
        print("[X] Error: Faltan archivos CSV. Debes correr los modos 'pi' y 'sac' primero.")
        return

    df_pi = pd.read_csv("stress_data_pi.csv")
    df_sac = pd.read_csv("stress_data_sac.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(df_pi["Time(s)"], df_pi["Reference(C)"], 'k--', linewidth=2, label="Referencia")
    ax1.plot(df_pi["Time(s)"], df_pi["Temperature(C)"], color='#d62728', linewidth=2.5, alpha=0.85, label="PI Clásico")
    ax1.plot(df_sac["Time(s)"], df_sac["Temperature(C)"], color='#1f77b4', linewidth=2.5, alpha=0.9, label="SAC Puro (Propuesto)")
    
    ax1.set_ylabel("Temperatura (°C)", fontsize=12, fontweight='bold')
    ax1.set_title("Prueba de Estrés Térmico en Hardware: PI vs SAC Puro", fontsize=14, fontweight='bold')
    ax1.legend(loc="upper right", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.7)

    ax2.plot(df_pi["Time(s)"], df_pi["Control_Effort(%)"], color='#d62728', alpha=0.6, drawstyle='steps-post', label="Qu PI (%)")
    ax2.plot(df_sac["Time(s)"], df_sac["Control_Effort(%)"], color='#1f77b4', alpha=0.8, drawstyle='steps-post', label="Qu SAC (%)")
    
    ax2.set_xlabel("Tiempo (s)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Potencia (%)", fontsize=12, fontweight='bold')
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig("resultado_final_monografia.png", dpi=300)
    print("[OK] Gráfica guardada como 'resultado_final_monografia.png'")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["pi", "sac", "plot"])
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "pi":
        ejecutar_prueba(modo="pi")
    elif args.mode == "sac":
        ejecutar_prueba(modo="sac", model_path=args.model)
    elif args.mode == "plot":
        graficar_comparativa()