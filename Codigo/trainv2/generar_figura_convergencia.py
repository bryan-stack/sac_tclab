"""
generar_figura_convergencia.py — Curva de convergencia del entrenamiento SAC
============================================================================
Reconstruye la convergencia a partir de la recompensa de entrenamiento
(rollout/ep_rew_mean, media móvil de Stable-Baselines3 sobre las últimas 100
episodios) leída del registro de TensorBoard, y marca las transiciones del
Curriculum Learning (curriculum/stage). Esta señal es mucho más estable que
la recompensa de evaluación, que es ruidosa por la naturaleza estocástica de
los episodios de evaluación (Domain Randomization, tiempo muerto y arranques
en frío aleatorios).

Fuente: logs/sac_tclab_v4/SAC_1  (eventos de TensorBoard)
Salida: figuras_articulo/fig11_convergencia.png/.pdf
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = "./figuras_articulo"
os.makedirs(OUT, exist_ok=True)
TB_DIR = "logs/sac_tclab_v4/SAC_1"

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "legend.fontsize": 9.5, "figure.dpi": 150,
    "savefig.dpi": 400, "savefig.bbox": "tight", "axes.grid": True,
    "grid.linestyle": ":", "grid.alpha": 0.6,
})

STAGE_BG = ["#EBF5FB", "#E8F8F0", "#FEF5E7", "#FDEDEC"]   # fondo por etapa


def rolling(x, w=11):
    n = len(x); out = np.empty(n); h = w // 2
    for i in range(n):
        out[i] = x[max(0, i - h):min(n, i + h + 1)].mean()
    return out


def load_tb():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(TB_DIR); acc.Reload()
    rew = acc.Scalars("rollout/ep_rew_mean")
    s_r = np.array([e.step for e in rew], float)
    v_r = np.array([e.value for e in rew], float)
    stage_tag = "curriculum/stage" if "curriculum/stage" in acc.Tags()["scalars"] else "custom/stage"
    st = acc.Scalars(stage_tag)
    s_s = np.array([e.step for e in st], float)
    v_s = np.array([e.value for e in st], float)
    # transiciones: primer paso en que la etapa alcanza 2, 3, 4
    trans = {}
    for stage in (2, 3, 4):
        idx = np.where(v_s >= stage)[0]
        if len(idx):
            trans[stage] = s_s[idx[0]]
    return s_r, v_r, trans, s_s.max()


def main():
    try:
        steps, rew, trans, smax = load_tb()
    except Exception as e:
        print("[X] No se pudo leer TensorBoard:", repr(e))
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    # Fondos por etapa
    bounds = [0] + [trans[s] for s in (2, 3, 4) if s in trans] + [steps.max()]
    for i in range(len(bounds) - 1):
        ax.axvspan(bounds[i], bounds[i + 1], color=STAGE_BG[i % 4], alpha=0.45, zorder=0)

    ax.plot(steps, rew, color="#9ecae1", lw=1.0, alpha=0.7, label="Recompensa de entrenamiento")
    ax.plot(steps, rolling(rew), color="#08519c", lw=2.4, label="Media móvil")

    # Líneas y etiquetas de transición de etapa
    ymin = rew.min()
    for s in (2, 3, 4):
        if s in trans:
            ax.axvline(trans[s], color="#555555", ls="--", lw=1.0, alpha=0.8)
            ax.text(trans[s], ymin, f" Etapa {s}", rotation=90, va="bottom", ha="right",
                    fontsize=8.5, color="#555555")
    # Etiqueta de Etapa 1
    ax.text(bounds[0], ymin, " Etapa 1", rotation=90, va="bottom", ha="left",
            fontsize=8.5, color="#555555")

    ax.set_xlabel("Pasos de entrenamiento")
    ax.set_ylabel("Recompensa media de episodio")
    ax.set_title("Convergencia del Entrenamiento del Agente SAC", fontweight="bold")
    ax.xaxis.set_major_locator(MultipleLocator(250_000))
    ax.legend(loc="lower right", framealpha=0.9)
    ax.margins(x=0.01)

    plt.savefig(f"{OUT}/fig11_convergencia.pdf")
    plt.savefig(f"{OUT}/fig11_convergencia.png")
    plt.close()
    print(f"[OK] fig11_convergencia guardada en {OUT}/")
    print(f"     ep_rew_mean: inicio={rew[0]:.0f}  final={rew[-1]:.0f}  max={rew.max():.0f}")
    print(f"     transiciones de etapa (pasos): {{k: int(v) for k,v in trans.items()}}".replace("trans",""))
    for s, v in trans.items():
        print(f"       Etapa {s} desde paso {int(v)}")


if __name__ == "__main__":
    main()
