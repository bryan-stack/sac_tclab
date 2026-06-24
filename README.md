# Control de Temperatura en TCLab con SAC + Curriculum Learning

Material de la monografía: control robusto de temperatura en una planta de
Primer Orden con Tiempo Muerto (TCLab Arduino) mediante un agente Soft
Actor-Critic entrenado con Curriculum Learning, comparado en hardware real
contra un PI clásico, un NMPC puro y una arquitectura híbrida residual NMPC-SAC.

## Contenido

- **Manuscrito/**
  - `articulo_mdpi_sac_tclab.pdf` — artículo final.
  - `articulo_mdpi_sac_tclab.tex` — fuente LaTeX (compila con pdfLaTeX).
  - `Definitions/` — plantilla MDPI, necesaria para compilar.
  - `figuras/` — todas las figuras en alta resolución (PNG y PDF, ≥400 dpi).
- **Codigo/**
  - `trainv2/` — entorno y agente SAC puro (observación de 46 dimensiones),
    entrenamiento, pruebas, métricas y generación de figuras.
  - `nmpc/` — solver NMPC, entorno y agente residual NMPC-SAC (56 dimensiones).
- **Referencia/** — paper base (Soza Mamani & Prado Romo, *Processes* 2025).

Cada carpeta de `Codigo/` incluye, junto a los scripts, `models/` (el mejor
modelo y el final de cada agente), `logs/` (registros de evaluación usados para
las curvas de convergencia) y los `.csv` de hardware y simulación del artículo.

## Scripts principales

`trainv2/`
- `tclab_env.py` — entorno Gymnasium de la planta (ODE no lineal, tiempo muerto,
  ruido de ADC y Domain Randomization) con las cuatro etapas del curriculum.
- `train_sac_v3.py` — entrenamiento del SAC.
- `stress_suite_v2.py` — prueba de estrés de 600 s (PI / NMPC / SAC).
- `setpoint_const.py` — prueba de regulación a setpoint constante.
- `nmpc_sensitivity.py` — sensibilidad del NMPC al horizonte y a los pesos.
- `calcular_metricas.py`, `agregar_metricas.py`, `analisis_completo.py` — métricas.
- `generar_figuras_4ctrl.py`, `generar_figura_setpoint_4ctrl.py`,
  `generar_figura_convergencia.py`, `generar_figuras.py` — figuras del artículo.

`nmpc/`
- `nmpc_solver.py` — solver NMPC en horizonte deslizante (SLSQP).
- `tclab_env.py` — entorno de la arquitectura residual NMPC-SAC.
- `train_nmpc_sac.py` — entrenamiento del residuo SAC.
- `stress_nmpcsac.py` — prueba de estrés del NMPC-SAC.

## Requisitos

Python 3.12 con numpy, scipy, pandas, matplotlib, gymnasium, stable-baselines3
y torch. La librería `tclab` solo hace falta para correr sobre el hardware real;
todas las pruebas aceptan la opción `--sim` para validarse contra el modelo ODE.

## Reproducir

Compilar el artículo:

    cd Manuscrito
    pdflatex articulo_mdpi_sac_tclab.tex   # ejecutar 3 veces (referencias)

Validar una prueba en simulación (sin hardware):

    cd Codigo/trainv2
    python stress_suite_v2.py --mode sac --reps 1 --sim --model models/sac_tclab_v4_best/best_model

Regenerar las figuras del artículo:

    cd Codigo/trainv2
    python generar_figuras_4ctrl.py
    python generar_figura_setpoint_4ctrl.py
    python generar_figura_convergencia.py
