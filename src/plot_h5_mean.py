import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.build_h5_dataset import build_h5_dataset


def plot_mean_spectrum_per_organ(freq, X, y, organ_mapping):
    """
    freq: (B,)
    X: (N, B)
    y: (N,)
    organ_mapping: dict {organ: label}
    """

    plt.figure(figsize=(10, 5))

    # label -> organ name
    label_to_organ = {v: k for k, v in organ_mapping.items()}

    for label in sorted(label_to_organ.keys()):
        organ = label_to_organ[label]

        mask = (y == label)
        if np.sum(mask) == 0:
            continue

        mean_spectrum = X[mask].mean(axis=0)

        plt.plot(freq, mean_spectrum, label=organ)

    plt.xlabel("Frequency (THz)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("Mean Spectrum per Organ")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    freq, X, y, groups, days, cond, organ_mapping, _ = build_h5_dataset(
        "metadata/metadata_h5.csv"
    )

    print("Frequency range:", freq.min(), freq.max())
    print("Number of bands:", len(freq))

    plot_mean_spectrum_per_organ(freq, X, y, organ_mapping)