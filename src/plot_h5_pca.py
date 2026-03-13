from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =========================
# Settings from environment
# =========================
DATA_DIR = Path("artifacts") / "training_h5_2"

THRESH = os.environ.get("HSI_THRESH", "85")
INNER = os.environ.get("HSI_INNER", "40")
PIXELS = os.environ.get("HSI_PIXELS", "2000")

FREQ_MIN = float(os.environ.get("HSI_FMIN", "333"))
FREQ_MAX = float(os.environ.get("HSI_FMAX", "748"))

N_SAMPLE = int(os.environ.get("HSI_PCA_N", "5000"))
RANDOM_SEED = int(os.environ.get("HSI_PCA_SEED", "42"))

SUFFIX = f"t{THRESH}_i{INNER}_p{PIXELS}"


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
        raise ValueError(f"No bands found in range [{fmin}, {fmax}]")
    return freq[mask], X[:, mask]


def sample_data(X, y, days, n_sample, seed=42):
    rng = np.random.default_rng(seed)
    n_total = X.shape[0]
    n_use = min(n_sample, n_total)
    idx = rng.choice(n_total, size=n_use, replace=False)
    return X[idx], y[idx], days[idx], idx


def label_to_name_mapping(mapping_df):
    return dict(zip(mapping_df["label"], mapping_df["organ"]))


def make_scatter_plot(X_pca, y_sample, mapping_df, out_path, title):
    label_to_name = label_to_name_mapping(mapping_df)

    unique_labels = sorted(np.unique(y_sample))
    plt.figure(figsize=(9, 7))

    for lab in unique_labels:
        mask = y_sample == lab
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=10,
            alpha=0.55,
            label=label_to_name[lab]
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(markerscale=1.5, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_day_scatter_plot(X_pca, days_sample, out_path, title):
    unique_days = sorted(np.unique(days_sample))
    plt.figure(figsize=(9, 7))

    for d in unique_days:
        mask = days_sample == d
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=10,
            alpha=0.55,
            label=f"Day {d}"
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(markerscale=1.5, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_loading_plot(freq_sub, pca, out_path, title):
    plt.figure(figsize=(10, 5))
    plt.plot(freq_sub, pca.components_[0], label="PC1 loading")
    plt.plot(freq_sub, pca.components_[1], label="PC2 loading")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Loading weight")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    print(f"Loading dataset suffix: {SUFFIX}")
    freq, X, y, days, mapping = load_dataset()

    print("Original shapes:")
    print("  freq:", freq.shape)
    print("  X   :", X.shape)
    print("  y   :", y.shape)
    print("  days:", days.shape)

    freq_sub, X_sub = apply_frequency_range(freq, X, FREQ_MIN, FREQ_MAX)

    print("\nAfter frequency selection:")
    print("  freq_sub:", freq_sub.shape)
    print("  X_sub   :", X_sub.shape)
    print("  first/last freq:", freq_sub[0], freq_sub[-1])

    X_sample, y_sample, days_sample, idx = sample_data(
        X_sub, y, days, N_SAMPLE, seed=RANDOM_SEED
    )

    print("\nAfter random sampling:")
    print("  X_sample:", X_sample.shape)
    print("  y_sample:", y_sample.shape)

    # Standardize before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)

    print("\nExplained variance ratio:")
    print("  PC1:", pca.explained_variance_ratio_[0])
    print("  PC2:", pca.explained_variance_ratio_[1])
    print("  PC1 + PC2:", pca.explained_variance_ratio_.sum())

    out_prefix = f"{SUFFIX}_f{int(FREQ_MIN)}_{int(FREQ_MAX)}_n{X_sample.shape[0]}"

    # 1) organ-colored PCA
    scatter_out = DATA_DIR / f"pca_scatter_organ_{out_prefix}.png"
    make_scatter_plot(
        X_pca,
        y_sample,
        mapping,
        scatter_out,
        title=f"PCA of H5 spectra by organ\n{SUFFIX}, freq [{FREQ_MIN}, {FREQ_MAX}], n={X_sample.shape[0]}"
    )

    # 2) day-colored PCA
    day_scatter_out = DATA_DIR / f"pca_scatter_day_{out_prefix}.png"
    make_day_scatter_plot(
        X_pca,
        days_sample,
        day_scatter_out,
        title=f"PCA of H5 spectra by day\n{SUFFIX}, freq [{FREQ_MIN}, {FREQ_MAX}], n={X_sample.shape[0]}"
    )

    # 3) loading plot
    loading_out = DATA_DIR / f"pca_loading_{out_prefix}.png"
    make_loading_plot(
        freq_sub,
        pca,
        loading_out,
        title=f"PCA loading plot\n{SUFFIX}, freq [{FREQ_MIN}, {FREQ_MAX}]"
    )

    # 4) explained variance csv
    var_df = pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_
    })
    var_csv = DATA_DIR / f"pca_variance_{out_prefix}.csv"
    var_df.to_csv(var_csv, index=False)

    print("\nSaved files:")
    print(" ", scatter_out)
    print(" ", day_scatter_out)
    print(" ", loading_out)
    print(" ", var_csv)


if __name__ == "__main__":
    main()