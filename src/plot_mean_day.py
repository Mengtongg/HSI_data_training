import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("artifacts") / "training_h5_2"

# ===== extraction parameter =====
THRESH = os.environ.get("HSI_THRESH", "85")
INNER = os.environ.get("HSI_INNER", "40")
PIXELS = os.environ.get("HSI_PIXELS", "2000")

SUFFIX = f"t{THRESH}_i{INNER}_p{PIXELS}"

# ===== frequency range =====
FREQ_MIN = float(os.environ.get("HSI_FMIN", "333"))
FREQ_MAX = float(os.environ.get("HSI_FMAX", "748"))


def load_dataset():
    freq = np.load(DATA_DIR / f"freq_h5_{SUFFIX}.npy")
    X = np.load(DATA_DIR / f"X_h5_{SUFFIX}.npy")
    y = np.load(DATA_DIR / f"y_h5_{SUFFIX}.npy")
    days = np.load(DATA_DIR / f"days_h5_{SUFFIX}.npy")
    mapping = pd.read_csv(DATA_DIR / f"organ_mapping_h5_{SUFFIX}.csv")

    return freq, X, y, days, mapping


def apply_frequency_range(freq, X, fmin, fmax):
    mask = (freq >= fmin) & (freq <= fmax)

    if mask.sum() == 0:
        raise ValueError(f"No frequency bands found in range [{fmin}, {fmax}]")

    freq_sub = freq[mask]
    X_sub = X[:, mask]

    return freq_sub, X_sub


def plot_mean_spectrum_per_day(freq, X, y, days, mapping, save_path=None):
    """
    Create a 1x3 grid:
    panel 1 -> Day 1 mean spectra per organ
    panel 2 -> Day 2 mean spectra per organ
    panel 3 -> Day 3 mean spectra per organ
    """
    label_to_organ = dict(zip(mapping["label"], mapping["organ"]))
    organ_labels = sorted(np.unique(y))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, day in zip(axes, [1, 2, 3]):
        day_mask = days == day

        for label in organ_labels:
            organ_mask = y == label
            mask = day_mask & organ_mask

            if mask.sum() == 0:
                continue

            mean_spectrum = X[mask].mean(axis=0)
            organ_name = label_to_organ[label]

            ax.plot(freq, mean_spectrum, label=organ_name)

        ax.set_title(f"Day {day}")
        ax.set_xlabel("Frequency (THz)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean Intensity (a.u.)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Mean Spectrum per Organ by Day", fontsize=16)
   
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def main():
    freq, X, y, days, mapping = load_dataset()

    print("Original dataset loaded:")
    print("  freq shape :", freq.shape)
    print("  X shape    :", X.shape)
    print("  y shape    :", y.shape)
    print("  days shape :", days.shape)
    print("  unique days:", np.unique(days))

    freq_sub, X_sub = apply_frequency_range(freq, X, FREQ_MIN, FREQ_MAX)

    print("\nAfter frequency selection:")
    print("  freq_sub shape:", freq_sub.shape)
    print("  X_sub shape   :", X_sub.shape)
    print("  first freq    :", freq_sub[0])
    print("  last freq     :", freq_sub[-1])

    save_name = f"mean_spectrum_by_day_{SUFFIX}_f{int(FREQ_MIN)}_{int(FREQ_MAX)}.png"
    save_path = DATA_DIR / save_name

    plot_mean_spectrum_per_day(
        freq_sub,
        X_sub,
        y,
        days,
        mapping,
        save_path=save_path
    )

    print(f"\nSaved figure to: {save_path}")


if __name__ == "__main__":
    main()