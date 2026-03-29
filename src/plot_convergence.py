from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.build_h5_dataset import build_h5_dataset


META_CSV = "metadata/metadata_h5.csv"

# fixed parameters
THRESHOLD_PERCENTILE = 85
INNER_PERCENTILE = 40
FREQ_MIN = 333
FREQ_MAX = 748
RANDOM_SEED = 42
MAX_ITER = 4000

# test these sampling sizes
PIXEL_SETTINGS = [100, 250, 500, 750, 1000, 1500, 2000, 3000]


def apply_frequency_range(freq, X, fmin, fmax):
    mask = (freq >= fmin) & (freq <= fmax)
    if mask.sum() == 0:
        raise ValueError(f"No frequency bands found in range [{fmin}, {fmax}]")
    return freq[mask], X[:, mask]


def apply_gradient_second_derivative(X, freq):
    X_first = np.gradient(X, freq, axis=1)
    X_second = np.gradient(X_first, freq, axis=1)
    return X_second


def majority_vote_per_file(y_pred, groups):
    votes = {}
    for pred, gid in zip(y_pred, groups):
        votes.setdefault(gid, []).append(pred)

    out = {}
    for gid, preds in votes.items():
        vals, counts = np.unique(preds, return_counts=True)
        out[gid] = vals[np.argmax(counts)]
    return out


def compute_metrics_for_setting(pixels_per_file):
    freq, X, y, groups, days, cond, organ_to_label, file_stats_df = build_h5_dataset(
        meta_csv=META_CSV,
        threshold_percentile=THRESHOLD_PERCENTILE,
        inner_percentile=INNER_PERCENTILE,
        pixels_per_file=pixels_per_file,
        random_seed=RANDOM_SEED,
    )

    freq_sub, X_sub = apply_frequency_range(freq, X, FREQ_MIN, FREQ_MAX)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=MAX_ITER))
    ])

    spectrum_accs = []
    macro_f1s = []
    file_accs = []

    for test_day in [1, 2, 3]:
        train_mask = days != test_day
        test_mask = days == test_day

        X_train_raw = X_sub[train_mask]
        y_train = y[train_mask]

        X_test_raw = X_sub[test_mask]
        y_test = y[test_mask]
        groups_test = groups[test_mask]

        X_train = apply_gradient_second_derivative(X_train_raw, freq_sub)
        X_test = apply_gradient_second_derivative(X_test_raw, freq_sub)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # spectrum-level
        sacc = accuracy_score(y_test, y_pred)
        mf1 = f1_score(y_test, y_pred, average="macro")

        # file-level
        file_pred = majority_vote_per_file(y_pred, groups_test)

        file_true = {}
        for yt, gid in zip(y_test, groups_test):
            file_true.setdefault(gid, yt)

        file_ids = sorted(file_true.keys())
        y_file_true = np.array([file_true[fid] for fid in file_ids])
        y_file_pred = np.array([file_pred[fid] for fid in file_ids])

        facc = accuracy_score(y_file_true, y_file_pred)

        spectrum_accs.append(sacc)
        macro_f1s.append(mf1)
        file_accs.append(facc)

    return {
        "pixels_per_file": pixels_per_file,

        "day1_spectrum_acc": spectrum_accs[0],
        "day2_spectrum_acc": spectrum_accs[1],
        "day3_spectrum_acc": spectrum_accs[2],
        "mean_spectrum_acc": np.mean(spectrum_accs),

        "day1_macro_f1": macro_f1s[0],
        "day2_macro_f1": macro_f1s[1],
        "day3_macro_f1": macro_f1s[2],
        "mean_macro_f1": np.mean(macro_f1s),

        "day1_file_acc": file_accs[0],
        "day2_file_acc": file_accs[1],
        "day3_file_acc": file_accs[2],
        "mean_file_acc": np.mean(file_accs),
    }

def plot_convergence(results_df, y_col, ylabel, title):
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["pixels_per_file"], results_df[y_col], marker="o")

    for x, y in zip(results_df["pixels_per_file"], results_df[y_col]):
        plt.text(x, y + 0.01, f"{y:.3f}", ha="center", fontsize=8)

    plt.xlabel("Sampled spectra per file")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    rows = []

    for pixels_per_file in PIXEL_SETTINGS:
        print(f"\nRunning pixels_per_file = {pixels_per_file}")
        row = compute_metrics_for_setting(pixels_per_file)
        rows.append(row)

    results_df = pd.DataFrame(rows)

    print("\nResults:")
    print(results_df)

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    results_df.to_csv(out_dir / "file_accuracy_convergence.csv", index=False)
    plot_convergence(
        results_df,
        y_col="mean_file_acc",
        ylabel="Mean file accuracy",
        title="Convergence of file-level accuracy with pixel sampling number"
    )

    plot_convergence(
        results_df,
        y_col="mean_spectrum_acc",
        ylabel="Mean spectrum accuracy",
        title="Convergence of spectrum-level accuracy with pixel sampling number"
    )

    plot_convergence(
        results_df,
        y_col="mean_macro_f1",
        ylabel="Mean macro F1",
        title="Convergence of macro F1 with pixel sampling number"
    )

if __name__ == "__main__":
    main()
