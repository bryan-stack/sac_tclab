"""
test_hardware_nmpc_sac.py — Prueba física de la Arquitectura Residual en TCLab
==============================================================================
Este script despliega el modelo híbrido definitivo en el hardware.
Combina el optimizador matemático (NMPC) con la red neuronal (SAC) 
en tiempo real utilizando los datos del sensor físico.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import tclab
from tclab import clock

# Importamos el entorno híbrido que ya tiene instanciado el NMPCSolver
from tclab_env import TCLabEnv

class NMPCHardwareAdapter(TCLabEnv):
    """
    Adaptador que hereda del entorno residual.
    Sustituye la integración de Euler (ODE) por la lectura del hardware.
    """
    def __init__(self, stage=1):
        super().__init__(stage=stage)

    def reset_hardware(self, real_T0):
        super().reset()
        # Sincronizamos la temperatura inicial
        self.T_K = real_T0 + 273.15
        self.Ta_C = real_T0
        self.eT_prev = self.T_ref_C - real_T0
        self.eT_integral = 0.0
        
        # Reiniciar variables del control residual
        self._action_history = [0.0] * self.HISTORY_LEN
        self.Q_NMPC_prev = 30.0
        self.Q_NMPC_cached = 30.0
        self.step_count = 0
        
        # Sincronizar el ambiente del NMPC con la temperatura real inicial
        self._params["Ta"] = self.T_K
        self.nmpc.update_params(**self._params)
        
        return self._build_obs(real_T0, self.eT_prev, 0.0)

    def step_hardware(self, action, real_T):
        idx = min(self.step_count, len(self.T_ref_trajectory) - 1)
        self.T_ref_C = float(self.T_ref_trajectory[idx])

        # 1. Ejecutar Experto NMPC con cacheo (refresco cada N pasos)
        if self.step_count % self.NMPC_REFRESH_STEPS == 0:
            Q_NMPC = self.nmpc.compute_action(
                T0_C    = real_T,
                T_ref_C = self.T_ref_C,
                Q_warm  = self.Q_NMPC_prev,
            )
            self.Q_NMPC_cached = Q_NMPC
            self.Q_NMPC_prev   = Q_NMPC
        else:
            Q_NMPC = self.Q_NMPC_cached

        # 2. Calcular Corrección del SAC
        delta_Q_SAC = float(action[0]) * self.MAX_RESIDUAL
        
        # 3. Fusión Híbrida
        Qu = float(np.clip(Q_NMPC + delta_Q_SAC, 0.0, 100.0))

        # Actualizar historial para el siguiente paso
        self._action_history.append(float(action[0]))
        self._action_history.pop(0)

        # Cálculo de errores con la temperatura FÍSICA
        eT_real = self.T_ref_C - real_T
        eT_der  = (eT_real - self.eT_prev) / self.dt
        self.eT_integral += eT_real * self.dt
        self.eT_prev = eT_real

        self.step_count += 1
        obs = self._build_obs(real_T, eT_real, eT_der)

        return obs, Qu, Q_NMPC, delta_Q_SAC, self.T_ref_C, eT_real


def run_hardware_nmpc_sac(model_path, stage=1, duration=500):
    print(f"\n{'='*55}")
    print(" INICIANDO PRUEBA EN HARDWARE: NMPC-SAC RESIDUAL")
    print(f" Modelo : {model_path}")
    print(f" Stage  : {stage}")
    print(f"{'='*55}\n")

    # 1. Cargar el modelo SAC híbrido
    model = SAC.load(model_path, device="cpu")
    
    # 2. Inicializar adaptador
    env = NMPCHardwareAdapter(stage=stage)
    env.episode_len = duration

    # 3. Almacenamiento de métricas detalladas
    log_time = []
    log_Tref = []
    log_Treal = []
    log_Qu_total = []
    log_Q_NMPC = []
    log_delta_Q = []

    # 4. Conexión al hardware
    try:
        with tclab.TCLab() as lab:
            print("[OK] Placa TCLab conectada.")
            T0 = lab.T1
            print(f"[+] Temperatura ambiente detectada: {T0:.2f} °C")
            
            obs = env.reset_hardware(real_T0=T0)
            
            print("\nIniciando control híbrido (Presiona Ctrl+C para abortar)...\n")
            
            for t in clock(duration, env.dt):
                # Inferencia del SAC
                action, _ = model.predict(obs, deterministic=True)
                
                # Lectura real
                T_real = lab.T1
                
                # Ejecutar lógica híbrida
                obs, Qu, Q_NMPC, delta_Q, T_ref, eT = env.step_hardware(action, T_real)
                
                # Accionar hardware
                lab.Q1(Qu)
                
                # Guardar métricas
                log_time.append(t)
                log_Tref.append(T_ref)
                log_Treal.append(T_real)
                log_Qu_total.append(Qu)
                log_Q_NMPC.append(Q_NMPC)
                log_delta_Q.append(delta_Q)
                
                if t % 10 == 0:
                    print(f"t: {t:3.0f}s | T_ref: {T_ref:5.2f} °C | T_real: {T_real:5.2f} °C "
                          f"| Q_NMPC: {Q_NMPC:5.1f}% | Δ_SAC: {delta_Q:5.1f}% | Qu_Total: {Qu:5.1f}%")

    except KeyboardInterrupt:
        print("\n[!] Prueba abortada por el usuario. Apagando calentador...")
    except Exception as e:
        print(f"\n[X] Error crítico: {e}")
        return

    # 5. Exportar y Graficar
    df = pd.DataFrame({
        "Time(s)": log_time,
        "Reference(C)": log_Tref,
        "Temperature(C)": log_Treal,
        "Control_Total(%)": log_Qu_total,
        "Base_NMPC(%)": log_Q_NMPC,
        "Residual_SAC(%)": log_delta_Q
    })
    
    csv_name = f"hardware_nmpcsac_stage_{stage}.csv"
    df.to_csv(csv_name, index=False)
    print(f"\n[OK] Datos exportados a {csv_name}")

    # Gráfica de 3 paneles para mostrar la descomposición del control
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Panel 1: Temperatura
    ax1.plot(df["Time(s)"], df["Reference(C)"], 'k--', label="Referencia")
    ax1.plot(df["Time(s)"], df["Temperature(C)"], 'b-', linewidth=2, label="NMPC-SAC (Real)")
    ax1.set_ylabel("Temperatura (°C)")
    ax1.set_title(f"Prueba en Hardware - Control Residual (Stage {stage})")
    ax1.legend()
    ax1.grid(True)

    # Panel 2: Acción Base NMPC vs Total
    ax2.plot(df["Time(s)"], df["Base_NMPC(%)"], 'k:', alpha=0.6, drawstyle='steps-post', label="Base NMPC")
    ax2.plot(df["Time(s)"], df["Control_Total(%)"], 'r-', drawstyle='steps-post', label="Acción Final (NMPC+SAC)")
    ax2.set_ylabel("Potencia (%)")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    # Panel 3: Residuo Aislado
    ax3.fill_between(df["Time(s)"], 0, df["Residual_SAC(%)"], color='orange', alpha=0.4, step="post", label="Residuo SAC (±15%)")
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.set_ylim(-16, 16)
    ax3.set_xlabel("Tiempo (s)")
    ax3.set_ylabel("Δ Corrección (%)")
    ax3.legend(loc="upper right")
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(f"hardware_nmpcsac_stage_{stage}.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Ruta al modelo residual.zip")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--duration", type=int, default=500)
    args = parser.parse_args()

    run_hardware_nmpc_sac(args.model, args.stage, args.duration)