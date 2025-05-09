#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_directory(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def plot_pred_vs_true(predictions_csv, output_path):
    df = pd.read_csv(predictions_csv)
    y_true = df["y_true"]
    y_pred = df["y_pred"]

    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', label="ideal")
    plt.xlabel("Rzeczywista wartość")
    plt.ylabel("Predykcja")
    plt.title("Predykcje vs. rzeczywiste")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_residuals_hist(predictions_csv, output_path):
    df = pd.read_csv(predictions_csv)
    errors = df["y_pred"] - df["y_true"]

    plt.figure(figsize=(6,4))
    plt.hist(errors, bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel("Błąd (y_pred – y_true)")
    plt.ylabel("Liczba próbek")
    plt.title("Histogram reszt")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_weights(weights_csv, features_csv, output_path):
    w = pd.read_csv(weights_csv, header=None).iloc[:,0].values
    feature_names = ["bias"] + list(
        pd.read_csv(features_csv)
          .drop(columns=["Age at enrollment"])
          .columns
    )

    plt.figure(figsize=(8,5))
    plt.barh(feature_names, w)
    plt.xlabel("Współczynniki")
    plt.title("Wagi regresji (closed-form)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    results_dir = "reg_results"
    ensure_directory(results_dir)

    preds_csv    = os.path.join(results_dir, "predictions_closed_form.csv")
    weights_csv  = os.path.join(results_dir, "weights_closed_form.csv")
    features_csv = "formatted_dataset.csv"

    plot_pred_vs_true(preds_csv, os.path.join(results_dir, "plot_pred_vs_true.png"))
    plot_residuals_hist(preds_csv, os.path.join(results_dir, "plot_residuals_hist.png"))
    plot_weights(weights_csv, features_csv, os.path.join(results_dir, "plot_weights.png"))

    print("🔖 Pliki wykresów zapisano w:", results_dir)

if __name__ == "__main__":
    main()
