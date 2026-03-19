import os
from pathlib import Path

from matplotlib import cm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from scipy.signal import savgol_filter

from scipy import sparse
from scipy.sparse.linalg import spsolve

DATA_DIR = Path("artifacts") / "training_h5_2"

# ===== extraction parameter =====
THRESH = os.environ.get("HSI_THRESH", "85")
INNER = os.environ.get("HSI_INNER", "40")
PIXELS = os.environ.get("HSI_PIXELS", "2000")

SUFFIX = f"t{THRESH}_i{INNER}_p{PIXELS}"

# ===== frequency range =====
FREQ_MIN = float(os.environ.get("HSI_FMIN", "333"))
FREQ_MAX = float(os.environ.get("HSI_FMAX", "748"))

# ===== model parameter =====
MAX_ITER = 4000

# PCA params
# N_PCA = int(os.environ.get("HSI_PCA_COMP", "30"))

# ===== derivative parameter =====
# Options:
#   none         -> no derivative
#   gradient     -> np.gradient first derivative
#   savgol_deriv -> Savitzky-Golay first derivative
DERIV_METHOD = os.environ.get("HSI_DERIV", "gradient")

# Savitzky-Golay params (only used when DERIV_METHOD == "savgol_deriv")
SG_WINDOW = int(os.environ.get("HSI_SG_WINDOW", "11"))
SG_POLY = int(os.environ.get("HSI_SG_POLY", "2"))

# ===== baseline correction parameter =====
# Options:
#   none  -> no baseline correction
#   asls  -> asymmetric least squares baseline correction
BASELINE_METHOD = os.environ.get("HSI_BASELINE", "none")

# AsLS params
BASELINE_LAM = float(os.environ.get("HSI_BASELINE_LAM", "1000000"))
BASELINE_P = float(os.environ.get("HSI_BASELINE_P", "0.001"))
BASELINE_NITER = int(os.environ.get("HSI_BASELINE_NITER", "5"))


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

    freq_sub = freq[mask]
    X_sub = X[:, mask]

    return freq_sub, X_sub, mask.sum()

def baseline_asls_single(y, lam=1e7, p=0.01, niter=5):
    """
    Asymmetric Least Squares baseline correction for one spectrum.

    Parameters
    ----------
    y : np.ndarray
        Shape (n_bands,)
    lam : float
        Smoothness parameter. Larger -> smoother baseline.
    p : float
        Asymmetry parameter. Small p puts baseline below peaks.
    niter : int
        Number of iterations.
    """
    L = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L),format="csc")
    w = np.ones(L)

    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(L, L))
        Z = (W + lam * (D.T @ D)).tocsc()
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return y - z


def apply_baseline_correction(X, method="none", lam=1e6, p=0.001, niter=5):
    """
    Apply baseline correction row by row.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, n_bands)
    method : str
        'none' or 'asls'
    """
    if method == "none":
        return X

    if method == "asls":
        X_corr = np.empty_like(X, dtype=np.float64)
        for i in range(X.shape[0]):
            X_corr[i] = baseline_asls_single(X[i], lam=lam, p=p, niter=niter)
        return X_corr

    raise ValueError(f"Unknown baseline correction method: {method}")


def apply_derivative(X, freq, method="gradient", sg_window=11, sg_poly=2):
    """
    Apply derivative along the frequency axis.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, n_bands)
    freq : np.ndarray
        Shape (n_bands,)
    method : str
        'none', 'gradient', 'gradient2', 'savgol_deriv', or 'savgol_deriv2'
    sg_window : int
        Window length for Savitzky-Golay derivative
    sg_poly : int
        Polynomial order for Savitzky-Golay derivative
    """
    if method == "none":
        return X

    if method == "gradient":
        # first derivative
        X_deriv = np.gradient(X, freq, axis=1)
        return X_deriv

    if method == "gradient2":
        # second derivative
        X_first = np.gradient(X, freq, axis=1)
        X_second = np.gradient(X_first, freq, axis=1)
        return X_second

    if method in ["savgol_deriv", "savgol_deriv2"]:
        if sg_window % 2 == 0:
            raise ValueError("SG_WINDOW must be odd for Savitzky-Golay filter.")

        if sg_window <= sg_poly:
            raise ValueError("SG_WINDOW must be greater than SG_POLY.")

        # approximate frequency spacing
        delta = np.mean(np.diff(freq))

        deriv_order = 1 if method == "savgol_deriv" else 2

        X_deriv = savgol_filter(
            X,
            window_length=sg_window,
            polyorder=sg_poly,
            deriv=deriv_order,
            delta=delta,
            axis=1
        )
        return X_deriv

    raise ValueError(f"Unknown derivative method: {method}")

def plot_example_derivative(freq, X_raw, X_processed, n_examples=5):
    """
    Plot a few spectra before and after derivative for sanity check.
    """
    n_examples = min(n_examples, X_raw.shape[0])

    plt.figure(figsize=(10, 5))
    for i in range(n_examples):
        plt.plot(freq, X_raw[i], alpha=0.7, label=f"raw_{i}" if i == 0 else None)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Intensity")
    plt.title("Example Raw Spectra")
    plt.tight_layout()
    plt.show()

    if DERIV_METHOD == "none":
        ylabel = "Intensity"
    elif DERIV_METHOD in ["gradient", "savgol_deriv"]:
        ylabel = "First derivative"
    elif DERIV_METHOD in ["gradient2", "savgol_deriv2"]:
        ylabel = "Second derivative"
    else:
        ylabel = "Processed intensity"

    plt.figure(figsize=(10, 5))
    for i in range(n_examples):
        plt.plot(freq, X_processed[i], alpha=0.7, label=f"proc_{i}" if i == 0 else None)
    plt.xlabel("Frequency (THz)")
    plt.ylabel(ylabel)
    plt.title(f"Example Processed Spectra ({DERIV_METHOD})")
    plt.tight_layout()
    plt.show()


def run_a1(freq, X, y, groups, days, mapping):
    label_to_organ = dict(zip(mapping["label"], mapping["organ"]))

    clf = Pipeline([
        ("scaler", StandardScaler()),
        #("pca", PCA(n_components=N_PCA, random_state=42)),
        ("lr", LogisticRegression(max_iter=MAX_ITER))
    ])

    print("\n====================================")
    print(f"Dataset suffix: {SUFFIX}")
    print(f"Frequency range: [{FREQ_MIN}, {FREQ_MAX}]")
    print(f"Number of bands used: {X.shape[1]}")
    print(f"Derivative method: {DERIV_METHOD}")
    if DERIV_METHOD == "savgol_deriv":
        print(f"SG window={SG_WINDOW}, poly={SG_POLY}")
    print("Model: StandardScaler + LogisticRegression")  
    print(f"Baseline correction: {BASELINE_METHOD}")
    if BASELINE_METHOD == "asls":
        print(f"AsLS lambda={BASELINE_LAM}, p={BASELINE_P}, niter={BASELINE_NITER}")  
    # print(f"PCA components: {N_PCA}")
    print("====================================")

    summary_rows = []

    for test_day in [1, 2, 3]:
        train_mask = days != test_day
        test_mask = days == test_day

        X_train_raw, y_train = X[train_mask], y[train_mask]
        X_test_raw, y_test = X[test_mask], y[test_mask]
        groups_test = groups[test_mask]

        # SG smoothing
        # X_train = apply_sg(X_train)
        # X_test  = apply_sg(X_test)

        # ===== baseline correction =====
        X_train_base = apply_baseline_correction(
            X_train_raw,
            method=BASELINE_METHOD,
            lam=BASELINE_LAM,
            p=BASELINE_P,
            niter=BASELINE_NITER
        )
        X_test_base = apply_baseline_correction(
            X_test_raw,
            method=BASELINE_METHOD,
            lam=BASELINE_LAM,
            p=BASELINE_P,
            niter=BASELINE_NITER
        )

        # ===== derivative transformation =====
        X_train = apply_derivative(
            X_train_base,
            freq,
            method=DERIV_METHOD,
            sg_window=SG_WINDOW,
            sg_poly=SG_POLY
        )
        X_test = apply_derivative(
            X_test_base,
            freq,
            method=DERIV_METHOD,
            sg_window=SG_WINDOW,
            sg_poly=SG_POLY
        )

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

        # ===== Confusion Matrix =====
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8,6))

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
            "derivative_method": DERIV_METHOD,
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

    freq_sub, X_sub, n_bands = apply_frequency_range(freq, X, FREQ_MIN, FREQ_MAX)

    print("\nAfter frequency selection:")
    print("  freq_sub shape:", freq_sub.shape)
    print("  X_sub shape   :", X_sub.shape)
    print("  first freq    :", freq_sub[0])
    print("  last freq     :", freq_sub[-1])
    print(f"  number of bands:", n_bands)

    # ===== quick sanity check plot =====
    X_preview_base = apply_baseline_correction(
        X_sub[:5],
        method=BASELINE_METHOD,
        lam=BASELINE_LAM,
        p=BASELINE_P,
        niter=BASELINE_NITER
    )

    X_preview = apply_derivative(
        X_preview_base,
        freq_sub,
        method=DERIV_METHOD,
        sg_window=SG_WINDOW,
        sg_poly=SG_POLY
    )
    plot_example_derivative(freq_sub, X_sub[:5], X_preview, n_examples=5)

    results = run_a1(freq_sub, X_sub, y, groups, days, mapping)

    print("\n===== Summary Table =====")
    print(results)

    # save summary
    summary_name = (
    f"summary_{SUFFIX}_f{int(FREQ_MIN)}_{int(FREQ_MAX)}"
    f"_base-{BASELINE_METHOD}_deriv-{DERIV_METHOD}.csv"
    )
    results.to_csv(DATA_DIR / summary_name, index=False)
    print(f"\nSaved summary to: {DATA_DIR / summary_name}")


if __name__ == "__main__":
    main()