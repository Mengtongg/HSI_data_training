import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# =========================
# Dataset location
# =========================
DATA_DIR = Path("artifacts") / "training_h5_2"

# =========================
# Dataset selection
# =========================
THRESH = os.environ.get("HSI_THRESH", "85")
INNER = os.environ.get("HSI_INNER", "40")
PIXELS = os.environ.get("HSI_PIXELS", "2000")

SUFFIX = f"t{THRESH}_i{INNER}_p{PIXELS}"

# =========================
# Frequency range
# =========================
FREQ_MIN = float(os.environ.get("HSI_FMIN", "333"))
FREQ_MAX = float(os.environ.get("HSI_FMAX", "748"))

# =========================
# SG smoothing params
# window length must be odd
# =========================
SG_WINDOW = int(os.environ.get("HSI_SG_WINDOW", "11"))
SG_POLY = int(os.environ.get("HSI_SG_POLY", "2"))

# =========================
# PCA params
# =========================
N_PCA = int(os.environ.get("HSI_PCA_COMP", "30"))

# =========================
# SVM params
# =========================
SVM_C = float(os.environ.get("HSI_SVM_C", "1.0"))
SVM_GAMMA = os.environ.get("HSI_SVM_GAMMA", "scale")


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


def apply_frequency_range(freq, X, fmin, fmax):
    mask = (freq >= fmin) & (freq <= fmax)
    if mask.sum() == 0:
        raise ValueError(f"No frequency bands found in range [{fmin}, {fmax}]")
    return freq[mask], X[:, mask]


def apply_sg_filter(X, window_length=11, polyorder=2):
    """
    Apply Savitzky-Golay smoothing to each spectrum.
    X shape: (N, B)
    """
    n_bands = X.shape[1]

    if window_length >= n_bands:
        # make it valid and odd
        window_length = n_bands - 1 if n_bands % 2 == 0 else n_bands
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 3:
        return X.copy()

    if polyorder >= window_length:
        polyorder = max(1, window_length - 2)

    return savgol_filter(
        X,
        window_length=window_length,
        polyorder=polyorder,
        axis=1
    )


def run_a1(freq, X, y, groups, days, mapping):
    label_to_organ = dict(zip(mapping["label"], mapping["organ"]))

    # IMPORTANT:
    # scaler + PCA are fitted on training only inside the pipeline
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=N_PCA, random_state=42)),
        ("svm", SVC(
            C=SVM_C,
            gamma=SVM_GAMMA,
            kernel="rbf"
        ))
    ])

    print("\n====================================")
    print(f"Dataset suffix: {SUFFIX}")
    print(f"Frequency range: [{FREQ_MIN}, {FREQ_MAX}]")
    print(f"Number of bands used: {X.shape[1]}")
    print(f"SG window={SG_WINDOW}, poly={SG_POLY}")
    print(f"PCA components={N_PCA}")
    print(f"SVM C={SVM_C}, gamma={SVM_GAMMA}")
    print("====================================")

    summary_rows = []

    for test_day in [1, 2, 3]:
        train_mask = days != test_day
        test_mask = days == test_day

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        groups_test = groups[test_mask]

        # SG smoothing is per-spectrum, so no data leakage issue
        X_train = apply_sg_filter(X_train, window_length=SG_WINDOW, polyorder=SG_POLY)
        X_test = apply_sg_filter(X_test, window_length=SG_WINDOW, polyorder=SG_POLY)

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

    freq_sub, X_sub = apply_frequency_range(freq, X, FREQ_MIN, FREQ_MAX)

    print("\nAfter frequency selection:")
    print("  freq_sub shape:", freq_sub.shape)
    print("  X_sub shape   :", X_sub.shape)
    print("  first freq    :", freq_sub[0])
    print("  last freq     :", freq_sub[-1])

    results = run_a1(freq_sub, X_sub, y, groups, days, mapping)

    print("\n===== Summary Table =====")
    print(results)

    out_name = (
        f"summary_sg_pca_svm_{SUFFIX}"
        f"_f{int(FREQ_MIN)}_{int(FREQ_MAX)}"
        f"_sg{SG_WINDOW}_{SG_POLY}"
        f"_pca{N_PCA}"
        f"_C{SVM_C}.csv"
    )
    results.to_csv(DATA_DIR / out_name, index=False)
    print(f"\nSaved summary to: {DATA_DIR / out_name}")


if __name__ == "__main__":
    main()