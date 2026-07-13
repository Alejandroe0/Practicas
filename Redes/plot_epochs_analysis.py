#!/usr/bin/env python3
"""
Script para graficar los resultados del análisis de épocas para múltiples redes.
Genera gráficos de Tiempo vs Épocas y Error vs Épocas.
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# Configuración
# ================================================================
OUT_DIR = "outs"
RESULTS_FILE = os.path.join(OUT_DIR, "epochs_analysis_all_models.csv")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ================================================================
# Cargar datos
# ================================================================
def load_results(filepath):
    """Carga los resultados del archivo CSV."""
    data = {}

    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            if model not in data:
                data[model] = {
                    "description": row["model_desc"],
                    "epochs": [],
                    "times": [],
                    "rmse": [],
                    "mae": [],
                }
            data[model]["epochs"].append(int(row["epochs"]))
            data[model]["times"].append(float(row["training_time"]))
            data[model]["rmse"].append(float(row["rmse"]))
            data[model]["mae"].append(float(row["mae"]))

    # Convertir a arrays numpy y ordenar por épocas
    for model in data:
        order = np.argsort(data[model]["epochs"])
        data[model]["epochs"] = np.array(data[model]["epochs"])[order]
        data[model]["times"] = np.array(data[model]["times"])[order]
        data[model]["rmse"] = np.array(data[model]["rmse"])[order]
        data[model]["mae"] = np.array(data[model]["mae"])[order]

    return data

# ================================================================
# Graficar
# ================================================================
def plot_results(data):
    """Genera gráficos comparativos de tiempo y error vs épocas."""

    # Colores para cada modelo
    colors = ["steelblue", "coral", "lightgreen", "mediumpurple", "gold"]
    markers = ["o", "s", "^", "D", "v"]

    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico 1: Tiempo vs Épocas
    for idx, (model_name, model_data) in enumerate(data.items()):
        ax1.plot(
            model_data["epochs"],
            model_data["times"],
            f"{markers[idx]}-",
            linewidth=2,
            markersize=8,
            label=model_data["description"],
            color=colors[idx],
        )

    ax1.set_xlabel("Número de épocas", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Tiempo de entrenamiento (s)", fontsize=12, fontweight="bold")
    ax1.set_title("Tiempo de entrenamiento vs Épocas", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.legend(fontsize=10, loc="best")

    # Gráfico 2: Error (RMSE y MAE) vs Épocas
    for idx, (model_name, model_data) in enumerate(data.items()):
        # RMSE con líneas sólidas
        ax2.plot(
            model_data["epochs"],
            model_data["rmse"],
            f"{markers[idx]}-",
            linewidth=2,
            markersize=6,
            label=f"{model_data['description']} (RMSE)",
            color=colors[idx],
        )

    ax2.set_xlabel("Número de épocas", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Error (RMSE)", fontsize=12, fontweight="bold")
    ax2.set_title("RMSE vs Épocas (todas las redes)", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")
    ax2.legend(fontsize=9, loc="best")

    plt.tight_layout()

    # Guardar figura
    plot_file = os.path.join(PLOT_DIR, "epochs_analysis_all_models.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"Gráfico guardado en: {plot_file}")

    # Crear segunda figura: solo RMSE por modelo (para claridad)
    fig2, axes = plt.subplots(1, len(data), figsize=(5*len(data), 5))
    if len(data) == 1:
        axes = [axes]

    for idx, (model_name, model_data) in enumerate(data.items()):
        ax = axes[idx]
        ax.plot(
            model_data["epochs"],
            model_data["rmse"],
            "o-",
            linewidth=2,
            markersize=8,
            color=colors[idx],
        )
        ax.plot(
            model_data["epochs"],
            model_data["mae"],
            "s--",
            linewidth=2,
            markersize=8,
            color=colors[idx],
            alpha=0.7,
        )
        ax.set_xlabel("Épocas", fontsize=11, fontweight="bold")
        ax.set_ylabel("Error", fontsize=11, fontweight="bold")
        ax.set_title(f"{model_data['description']}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.legend(["RMSE", "MAE"], fontsize=10)

    plt.tight_layout()
    plot_file2 = os.path.join(PLOT_DIR, "epochs_analysis_individual.png")
    plt.savefig(plot_file2, dpi=300, bbox_inches="tight")
    print(f"Gráfico individual guardado en: {plot_file2}")

    plt.show()

# ================================================================
# Main
# ================================================================
def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} no encontrado.")
        print("Ejecuta primero: conda run -n practicas python epochs_benchmark.py")
        return

    print(f"Cargando resultados desde: {RESULTS_FILE}")
    data = load_results(RESULTS_FILE)

    if not data:
        print("No se encontraron datos en el archivo.")
        return

    print(f"Datos cargados para {len(data)} modelo(s):")
    for model_name, model_data in data.items():
        print(f"  - {model_data['description']}: {len(model_data['epochs'])} puntos")

    print("\nGenerando gráficos...")
    plot_results(data)

if __name__ == "__main__":
    main()
