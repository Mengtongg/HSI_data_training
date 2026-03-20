import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

DATA_DIR = Path("artifacts") / "training_h5_2"

# ===== fixed best parameters =====
THRESH = os.environ.get("HSI_THRESH", "85")
INNER = os.environ.get("HSI_INNER", "40")
PIXELS = os.environ.get("HSI_PIXELS", "2000")

SUFFIX = f"t{THRESH}_i{INNER}_p{PIXELS}"

FREQ_MIN = float(os.environ.get("HSI_FMIN", "333"))
FREQ_MAX = float(os.environ.get("HSI_FMAX", "748"))

MAX_ITER = 4000


def majority_vote_per_file(y_pred, groups):
    votes = {}
    for pred, gid in zip(y_pred, groups):
        votes.setdefault(gid, []).append(pred)

    out = {}
    for gid, preds in votes.items():
        vals, counts = np.unique(preds, return_counts=True)
        out[gid] = vals[np.argmax(counts)]

    return out


def load_dataset():
    freq = np.load(DATA_DIR / f"freq_h5_{SUFFIX}.npy")
    X = np.load(DATA_DIR / f"X_h5_{SUFFIX}.npy")
    y = np.load(DATA_DIR / f"y_h5_{SUFFIX}.npy")
    groups = np.load(DATA_DIR / f"groups_h5_{SUFFIX}.npy", allow_pickle=True)
    days = np.load(DATA_DIR / f"days_h5_{SUFFIX}.npy")
    mapping = pd.read_csv(DATA_DIR / f"organ_mapping_h5_{SUFFIX}.csv")

    return freq, X, y, groups, days, mapping


def plot_class_counts(y, mapping):
    label_to_organ = dict(zip(mapping["label"], mapping["organ"]))

    unique_labels, counts = np.unique(y, return_counts=True)
    organ_names = [label_to_organ[l] for l in unique_labels]

    df = pd.DataFrame({
        "organ": organ_names,
        "count": counts
    }).sort_values("count", ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(df["organ"], df["count"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of spectra")
    plt.title("Number of spectra per organ (H5 dataset)")

    for i, v in enumerate(df["count"]):
        plt.text(i, v + max(df["count"]) * 0.01, str(v), ha="center", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_class_counts_by_day(y, days, mapping):
    label_to_organ = dict(zip(mapping["label"], mapping["organ"]))

    df = pd.DataFrame({
        "label": y,
        "day": days
    })
    df["organ"] = df["label"].map(label_to_organ)

    count_df = (
        df.groupby(["day", "organ"])
        .size()
        .reset_index(name="count")
    )

    organs = sorted(df["organ"].unique())
    day_values = sorted(df["day"].unique())

    x = np.arange(len(organs))
    width = 0.25

    plt.figure(figsize=(12, 5))

    for i, day in enumerate(day_values):
        sub = count_df[count_df["day"] == day].set_index("organ").reindex(organs, fill_value=0)
        plt.bar(x + i * width, sub["count"], width=width, label=f"Day {day}")

    plt.xticks(x + width, organs, rotation=45, ha="right")
    plt.ylabel("Number of spectra")
    plt.title("Number of spectra per organ by acquisition day")
    plt.legend()
    plt.tight_layout()

    plt.show()


def plot_scree_and_pca(X, y, days, mapping, title_prefix=""):
    label_to_organ = dict(zip(mapping["label"], mapping["organ"]))
    organ_names = np.array([label_to_organ[v] for v in y])

    pca = PCA(n_components=10, random_state=42)
    X_pca10 = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_ * 100

    pca2 = PCA(n_components=2, random_state=42)
    X_pca2 = pca2.fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Scree plot ---
    x = np.arange(1, len(explained) + 1)
    axes[0].bar(x, explained)
    axes[0].plot(x, explained, marker="o")
    for i, val in enumerate(explained):
        axes[0].text(x[i], val + 0.5, f"{val:.1f}%", ha="center", fontsize=8)
    axes[0].set_xlabel("Principal Components")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title(f"{title_prefix} Scree Plot")

    # --- PCA by day ---
    unique_days = sorted(np.unique(days))
    for d in unique_days:
        mask = days == d
        axes[1].scatter(
            X_pca2[mask, 0],
            X_pca2[mask, 1],
            s=8,
            alpha=0.6,
            label=f"Day {d}"
        )
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title(f"{title_prefix} PCA by Day")
    axes[1].legend()

    # --- PCA by organ ---
    unique_organs = np.unique(organ_names)
    for organ in unique_organs:
        mask = organ_names == organ
        axes[2].scatter(
            X_pca2[mask, 0],
            X_pca2[mask, 1],
            s=8,
            alpha=0.6,
            label=organ
        )
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    axes[2].set_title(f"{title_prefix} PCA by Organ")
    axes[2].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def apply_final_preprocessing(X, freq):
    # gradient second derivative
    X_first = np.gradient(X, freq, axis=1)
    X_second = np.gradient(X_first, freq, axis=1)

    # scaling
    scaler = StandardScaler()
    X_proc = scaler.fit_transform(X_second)

    return X_proc


def apply_frequency_range(freq, X, fmin, fmax):
    mask = (freq >= fmin) & (freq <= fmax)

    if mask.sum() == 0:
        raise ValueError(f"No frequency bands found in range [{fmin}, {fmax}]")

    freq_sub = freq[mask]
    X_sub = X[:, mask]

    return freq_sub, X_sub, mask.sum()


def apply_gradient_second_derivative(X, freq):
    """
    Apply second derivative along the frequency axis.
    """
    X_first = np.gradient(X, freq, axis=1)
    X_second = np.gradient(X_first, freq, axis=1)
    return X_second


def plot_example_processed_spectra(freq, X_raw, X_processed, n_examples=5):
    """
    Plot a few spectra before and after second derivative for sanity check.
    """
    n_examples = min(n_examples, X_raw.shape[0])

    plt.figure(figsize=(10, 5))
    for i in range(n_examples):
        plt.plot(freq, X_raw[i], alpha=0.7)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Intensity")
    plt.title("Example Raw Spectra")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    for i in range(n_examples):
        plt.plot(freq, X_processed[i], alpha=0.7)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Second derivative")
    plt.title("Example Processed Spectra (Gradient Second Derivative)")
    plt.tight_layout()
    plt.show()


def run_experiment(freq, X, y, groups, days, mapping):
    label_to_organ = dict(zip(mapping["label"], mapping["organ"]))

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=MAX_ITER))
    ])

    print("\n====================================")
    print(f"Dataset suffix: {SUFFIX}")
    print(f"Frequency range: [{FREQ_MIN}, {FREQ_MAX}]")
    print(f"Number of bands used: {X.shape[1]}")
    print("Preprocessing: Gradient second derivative + StandardScaler")
    print("Model: LogisticRegression")
    print("Validation: Leave-one-day-out")
    print("====================================")

    summary_rows = []

    for test_day in [1, 2, 3]:
        train_mask = days != test_day
        test_mask = days == test_day

        X_train_raw, y_train = X[train_mask], y[train_mask]
        X_test_raw, y_test = X[test_mask], y[test_mask]
        groups_test = groups[test_mask]

        # fixed best preprocessing
        X_train = apply_gradient_second_derivative(X_train_raw, freq)
        X_test = apply_gradient_second_derivative(X_test_raw, freq)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        sacc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        labels_sorted = sorted(np.unique(y_test))
        target_names = [label_to_organ[i] for i in labels_sorted]

        print("\n==============================")
        print(f"TEST DAY = {test_day}")
        print(f"Spectrum-level: acc={sacc:.3f}, macroF1={macro_f1:.3f}")
        print(classification_report(
            y_test,
            y_pred,
            labels=labels_sorted,
            target_names=target_names,
            zero_division=0
        ))

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names
        )
        plt.xlabel("Predicted Organ")
        plt.ylabel("True Organ")
        plt.title(f"Confusion Matrix (Test Day {test_day})")
        plt.tight_layout()
        plt.show()

        # file-level majority vote
        file_pred = majority_vote_per_file(y_pred, groups_test)

        file_true = {}
        for yt, gid in zip(y_test, groups_test):
            file_true.setdefault(gid, yt)

        file_ids = sorted(file_true.keys())
        y_file_true = np.array([file_true[fid] for fid in file_ids])
        y_file_pred = np.array([file_pred[fid] for fid in file_ids])

        facc = accuracy_score(y_file_true, y_file_pred)

        print(f"File-level: acc={facc:.3f} (n_files={len(file_ids)})")

        summary_rows.append({
            "test_day": test_day,
            "spectrum_acc": sacc,
            "macro_f1": macro_f1,
            "file_acc": facc,
        })

    return pd.DataFrame(summary_rows)


def main():
    freq, X, y, groups, days, mapping = load_dataset()

    print("Original dataset loaded:")
    print("  freq shape :", freq.shape)
    print("  X shape    :", X.shape)
    print("  y shape    :", y.shape)
    print("  unique days:", np.unique(days))

    plot_class_counts(y, mapping)
    plot_class_counts_by_day(y, days, mapping)

    freq_sub, X_sub, n_bands = apply_frequency_range(freq, X, FREQ_MIN, FREQ_MAX)

    print("\nAfter frequency selection:")
    print("  freq_sub shape :", freq_sub.shape)
    print("  X_sub shape    :", X_sub.shape)
    print("  first freq     :", freq_sub[0])
    print("  last freq      :", freq_sub[-1])
    print("  number of bands:", n_bands)

    plot_scree_and_pca(
        X_sub, y, days, mapping,
        title_prefix="Before preprocessing")

    X_after = apply_final_preprocessing(X_sub, freq_sub)

    plot_scree_and_pca(
        X_after, y, days, mapping,
        title_prefix="After preprocessing"
    )

    # quick sanity check plot
    X_preview = apply_gradient_second_derivative(X_sub[:5], freq_sub)
    plot_example_processed_spectra(freq_sub, X_sub[:5], X_preview, n_examples=5)

    results = run_experiment(freq_sub, X_sub, y, groups, days, mapping)

    print("\n===== Summary Table =====")
    print(results)

    summary_name = (
        f"summary_{SUFFIX}_f{int(FREQ_MIN)}_{int(FREQ_MAX)}"
        f"_gradient2_scaler_lr.csv"
    )
    results.to_csv(DATA_DIR / summary_name, index=False)
    print(f"\nSaved summary to: {DATA_DIR / summary_name}")


if __name__ == "__main__":
    main()