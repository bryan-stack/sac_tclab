"""
test_hardware_pid.py — Prueba física de Controlador PI Clásico en TCLab
=======================================================================
Este script ejecuta un controlador PI discreto sintonizado por 
IMC-Skogestad directamente en el hardware para compararlo 1:1 con el agente SAC.
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
import tclab
from tclab import clock

class PIController:
    """
    Controlador PI discreto sintonizado para la dinámica FOPDT del TCLab.
    Kc = 7.3, tauI = 160.3
    """
    def __init__(self, Kc: float = 7.3, tauI: float = 160.3, dt: float = 1.0):
        self.Kc   = Kc
        self.tauI = tauI
        self.dt   = dt
        self._integral = 0.0
        self.Qu_prev = 0.0

    def compute(self, T_ref: float, T_obs: float) -> float:
        # Cálculo del error
        e = T_ref - T_obs
        
        # Integración numérica (Euler hacia adelante)
        self._integral += e * self.dt
        
        # Ley de control PI
        Qu = self.Kc * (e + self._integral / self.tauI)
        
        # Saturación del actuador (0 a 100%)
        # Implementación básica de anti-windup: frenar integración si satura
        if Qu > 100.0:
            Qu = 100.0
            self._integral -= e * self.dt 
        elif Qu < 0.0:
            Qu = 0.0
            self._integral -= e * self.dt

        self.Qu_prev = Qu
        return Qu


def run_hardware_pid_test(duration=500, T_ref_target=40.8):
    print(f"\n{'='*50}")
    print(" INICIANDO PRUEBA EN HARDWARE: CONTROLADOR PI")
    print(f" Referencia : {T_ref_target} °C")
    print(f"{'='*50}\n")

    # 1. Inicializar controlador (dt = 1s)
    pi = PIController()
    
    # 2. Almacenamiento de métricas
    log_time = []
    log_Tref = []
    log_Treal = []
    log_Qu = []

    # 3. Conexión al hardware
    try:
        with tclab.TCLab() as lab:
            print("[OK] Placa TCLab conectada.")
            T0 = lab.T1
            print(f"[+] Temperatura inicial detectada: {T0:.2f} °C")
            print("\nIniciando control (Presiona Ctrl+C para abortar con seguridad)...\n")
            
            for t in clock(duration, pi.dt):
                # Leer sensor físico
                T_real = lab.T1
                
                # Para replicar tu gráfica, mantenemos la referencia constante
                T_ref = T_ref_target
                
                # Calcular acción del PI
                Qu = pi.compute(T_ref, T_real)
                
                # Inyectar potencia al calentador
                lab.Q1(Qu)
                
                # Guardar logs
                log_time.append(t)
                log_Tref.append(T_ref)
                log_Treal.append(T_real)
                log_Qu.append(Qu)
                
                # Monitor en consola
                if t % 10 == 0:
                    eT = T_ref - T_real
                    print(f"t: {t:3.0f}s | T_ref: {T_ref:5.2f} °C | T_real: {T_real:5.2f} °C | Error: {eT:5.2f} °C | Qu: {Qu:5.1f} %")

    except KeyboardInterrupt:
        print("\n[!] Prueba abortada por el usuario. Apagando calentador...")
    except Exception as e:
        print(f"\n[X] Error de conexión: {e}")
        return

    # 4. Guardar y Graficar
    df = pd.DataFrame({
        "Time(s)": log_time,
        "Reference(C)": log_Tref,
        "Temperature(C)": log_Treal,
        "Control_Effort(%)": log_Qu
    })
    
    csv_name = "hardware_pid_comparison.csv"
    df.to_csv(csv_name, index=False)
    print(f"\n[OK] Datos exportados exitosamente a {csv_name}")

    # Graficar con el mismo formato del SAC
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(df["Time(s)"], df["Reference(C)"], 'k--', label="Referencia")
    ax1.plot(df["Time(s)"], df["Temperature(C)"], 'g-', linewidth=2, label="PI Clásico (Real)")
    ax1.set_ylabel("Temperatura (°C)")
    ax1.set_title(f"Prueba en Hardware - Controlador PI Clásico")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(df["Time(s)"], df["Control_Effort(%)"], 'r-', drawstyle='steps-post')
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("Potencia Calentador (%)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("hardware_pid_comparison.png")
    plt.show()

if __name__ == "__main__":
    # La referencia se fija en 40.8°C para homologar con la prueba SAC de la imagen
    run_hardware_pid_test(duration=500, T_ref_target=40.8)