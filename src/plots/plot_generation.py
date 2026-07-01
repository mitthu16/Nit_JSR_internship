import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = PROJECT_ROOT / "results"

CSV_FILE = RESULT_DIR / "benchmark.csv"

df = pd.read_csv(CSV_FILE)

rounds = df["Round"]


def save_plot(column, ylabel, filename):

    plt.figure(figsize=(8, 5))

    plt.plot(
        rounds,
        df[column],
        marker="o",
        linewidth=2,
    )

    plt.title(column)

    plt.xlabel("Communication Round")

    plt.ylabel(ylabel)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(RESULT_DIR / filename)

    plt.close()


save_plot("Loss", "Loss", "loss.png")

save_plot("Accuracy", "Accuracy (%)", "accuracy.png")

save_plot("CPU", "CPU (%)", "cpu.png")

save_plot("RAM", "RAM (%)", "ram.png")

save_plot("TrainingTime", "Training Time (Seconds)", "training_time.png")

print("\nPlots Generated Successfully!")