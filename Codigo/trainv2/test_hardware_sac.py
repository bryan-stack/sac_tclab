"""
test_hardware_sac.py — Prueba física del Agente SAC en la placa TCLab
=====================================================================
Este script conecta el modelo SAC entrenado directamente al hardware.
Sustituye la ecuación diferencial (ODE) por la lectura en tiempo real
del sensor de temperatura físico (T1).
"""

import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import tclab
from tclab import clock

# Importamos tu entorno original para reciclar la lógica de observaciones
from tclab_env import TCLabEnv

class HardwareAdapter(TCLabEnv):
    """
    Adaptador que hereda de tu entorno pero omite la simulación física.
    """
    def __init__(self, stage=1):
        super().__init__(stage=stage)

    def reset_hardware(self, real_T0):
        super().reset()
        # Sincronizamos la temperatura inicial simulada con la real
        self.T_K = real_T0 + 273.15
        self.Ta_C = real_T0
        self.eT_prev = self.T_ref_C - real_T0
        self.eT_integral = 0.0
        # Inicializamos el historial de acciones en 0
        self._action_history = [0.0] * self.HISTORY_LEN
        
        return self._build_obs(real_T0, self.eT_prev, 0.0)

    def step_hardware(self, action, real_T):
        idx = min(self.step_count, len(self.T_ref_trajectory) - 1)
        self.T_ref_C = float(self.T_ref_trajectory[idx])

        # Decodificar acción de SAC [-1, 1] a potencia [0, 100]
        Qu = float(np.clip((action[0] + 1.0) * 50.0, 0.0, 100.0))

        # Actualizar historial
        self._action_history.append(float(action[0]))
        self._action_history.pop(0)

        # Cálculo de errores con la temperatura FÍSICA
        eT_real = self.T_ref_C - real_T
        eT_der  = (eT_real - self.eT_prev) / self.dt
        self.eT_integral += eT_real * self.dt
        self.eT_prev = eT_real

        self.step_count += 1
        obs = self._build_obs(real_T, eT_real, eT_der)

        return obs, Qu, self.T_ref_C, eT_real


def run_hardware_test(model_path, stage=1, duration=500):
    print(f"\n{'='*50}")
    print(" INICIANDO PRUEBA EN HARDWARE TCLAB")
    print(f" Modelo : {model_path}")
    print(f" Stage  : {stage} (Perfil de referencia)")
    print(f"{'='*50}\n")

    # 1. Cargar el modelo SAC puro
    # Usamos device='cpu' para evitar latencias de transferencia en inferencia paso a paso
    model = SAC.load(model_path, device="cpu")
    
    # 2. Inicializar adaptador
    env = HardwareAdapter(stage=stage)
    env.episode_len = duration

    # 3. Almacenamiento de métricas
    log_time = []
    log_Tref = []
    log_Treal = []
    log_Qu = []

    # 4. Conexión al hardware
    try:
        with tclab.TCLab() as lab:
            print("[OK] Placa TCLab conectada.")
            
            # Leer temperatura inicial
            T0 = lab.T1
            print(f"[+] Temperatura ambiente detectada: {T0:.2f} °C")
            
            # Resetear estado interno del agente
            obs = env.reset_hardware(real_T0=T0)
            
            print("\nIniciando control (Presiona Ctrl+C para abortar con seguridad)...\n")
            
            # Bucle de control estricto (tclab.clock garantiza sincronización a 1 seg)
            for t in clock(duration, env.dt):
                # Predicción de la red neuronal
                action, _ = model.predict(obs, deterministic=True)
                
                # Leer sensor físico
                T_real = lab.T1
                
                # Calcular siguiente estado y potencia
                obs, Qu, T_ref, eT = env.step_hardware(action, T_real)
                
                # Inyectar potencia al calentador físico
                lab.Q1(Qu)
                
                # Guardar logs
                log_time.append(t)
                log_Tref.append(T_ref)
                log_Treal.append(T_real)
                log_Qu.append(Qu)
                
                # Monitor en consola
                if t % 10 == 0:
                    print(f"t: {t:3.0f}s | T_ref: {T_ref:5.2f} °C | T_real: {T_real:5.2f} °C | Error: {eT:5.2f} °C | Qu: {Qu:5.1f} %")

    except KeyboardInterrupt:
        print("\n[!] Prueba abortada por el usuario. Apagando calentador...")
    except Exception as e:
        print(f"\n[X] Error de conexión: {e}")
        return

    # 5. Guardar y Graficar
    df = pd.DataFrame({
        "Time(s)": log_time,
        "Reference(C)": log_Tref,
        "Temperature(C)": log_Treal,
        "Control_Effort(%)": log_Qu
    })
    
    csv_name = f"hardware_sac_stage_{stage}.csv"
    df.to_csv(csv_name, index=False)
    print(f"\n[OK] Datos exportados exitosamente a {csv_name}")

    # Graficar
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(df["Time(s)"], df["Reference(C)"], 'k--', label="Referencia")
    ax1.plot(df["Time(s)"], df["Temperature(C)"], 'b-', linewidth=2, label="SAC Puro (Real)")
    ax1.set_ylabel("Temperatura (°C)")
    ax1.set_title(f"Prueba en Hardware - Agente SAC (Stage {stage})")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(df["Time(s)"], df["Control_Effort(%)"], 'r-', drawstyle='steps-post')
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("Potencia Calentador (%)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"hardware_sac_stage_{stage}.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Ruta al modelo (ej. ./models/sac_tclab_v4)")
    parser.add_argument("--stage", type=int, default=1, help="Perfil de trayectoria a probar (1 a 4)")
    parser.add_argument("--duration", type=int, default=500, help="Duración en segundos")
    args = parser.parse_args()

    run_hardware_test(args.model, args.stage, args.duration)