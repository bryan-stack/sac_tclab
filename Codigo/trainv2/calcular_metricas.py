"""
calcular_metricas.py — Análisis de Desempeño Cuantitativo
=========================================================
Calcula índices de error (ISE, IAE, ITAE), Overshoot y 
esfuerzo de control (Energía y TV) a partir de los datos físicos.
"""

import pandas as pd
import numpy as np

def calcular_metricas(archivo_csv, nombre_modelo, dt=1.0):
    try:
        df = pd.read_csv(archivo_csv)
    except FileNotFoundError:
        print(f"[X] No se encontró el archivo: {archivo_csv}")
        return None

    # Extraer vectores de datos
    t = df['Time(s)'].values
    ref = df['Reference(C)'].values
    y = df['Temperature(C)'].values
    u = df['Control_Effort(%)'].values

    # Cálculo del Error
    e = ref - y

    # 1. Métricas de Error Integrales
    ise = np.sum((e**2) * dt)          # Integral Square Error (penaliza errores grandes/overshoot)
    iae = np.sum(np.abs(e) * dt)       # Integral Absolute Error (error acumulado total)
    itae = np.sum(t * np.abs(e) * dt)  # Integral Time Absolute Error (penaliza error en estado estacionario)

    # 2. Overshoot Máximo (Sobreimpulso)
    # Calculamos la máxima temperatura por encima de su referencia en ese instante
    overshoot_array = y - ref
    max_overshoot_C = np.max(overshoot_array) if np.max(overshoot_array) > 0 else 0.0

    # 3. Métricas de Esfuerzo de Control
    energia_total = np.sum(u * dt)     # Energía total inyectada
    tv_control = np.sum(np.abs(np.diff(u))) # Total Variation (Mide el desgaste del actuador/chattering)

    return {
        "Controlador": nombre_modelo,
        "ISE": round(ise, 2),
        "IAE": round(iae, 2),
        "ITAE": round(itae, 2),
        "Overshoot Máx (°C)": round(max_overshoot_C, 2),
        "Energía Total (%)": round(energia_total, 2),
        "Variación Control (TV)": round(tv_control, 2)
    }

if __name__ == "__main__":
    print("\nCalculando métricas de desempeño en Hardware...\n")
    
    resultados = []
    
    # === ANALIZA LAS PRUEBAS DE ESTRÉS DINÁMICO ===
    # (Si corriste la suite de pruebas dinámica)
    res_pi_stress = calcular_metricas("stress_data_pi.csv", "PI Clásico (Estrés)")
    res_sac_stress = calcular_metricas("stress_data_sac.csv", "SAC Puro (Estrés)")
    
    # === ANALIZA LAS PRUEBAS DE SETPOINT CONSTANTE ===
    # (Tus primeros CSVs subidos)
    res_pi_const = calcular_metricas("hardware_pid_comparison.csv", "PI Clásico (Constante)")
    res_sac_const = calcular_metricas("hardware_sac_stage_4.csv", "SAC Puro (Constante)")

    for res in [res_pi_stress, res_sac_stress, res_pi_const, res_sac_const]:
        if res is not None:
            resultados.append(res)

    if resultados:
        # Crear DataFrame para visualización tipo tabla
        df_resultados = pd.DataFrame(resultados)
        df_resultados.set_index("Controlador", inplace=True)
        
        print("="*85)
        print(" TABLA DE RESULTADOS CUANTITATIVOS (LISTA PARA LA TESIS)")
        print("="*85)
        print(df_resultados.to_string())
        print("="*85)
        
        # Exportar a Excel/CSV por si lo necesitas para el documento
        df_resultados.to_csv("tabla_metricas_final.csv")
        print("\n[OK] Tabla exportada a 'tabla_metricas_final.csv'")