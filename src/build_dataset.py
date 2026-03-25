from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from configs.paths import DATA_ROOT
from src.loader import load_txt_spectra


def interp_spectra(freq_src, spectra_src, freq_tgt):
    """
    Interpolate spectra from freq_src to freq_tgt.
    freq_src: (B_src,)
    spectra_src: (N, B_src)
    freq_tgt: (B_tgt,)
    returns: (N, B_tgt)
    """
    out = np.empty((spectra_src.shape[0], len(freq_tgt)), dtype=np.float32)
    for i in range(spectra_src.shape[0]):
        out[i] = np.interp(freq_tgt, freq_src, spectra_src[i])
    return out


def compute_common_grid(meta_csv: str, atol=1e-4):
    """
    Compute a common frequency grid using the intersection of ranges.
    Uses the smallest step size among files, then builds a grid over the common range.
    """
    meta = pd.read_csv(meta_csv)

    mins = []
    maxs = []
    steps = []

    for _, row in meta.iterrows():
        full_path = DATA_ROOT / str(row["path"])
        freq, spectra = load_txt_spectra(str(full_path))

        mins.append(freq.min())
        maxs.append(freq.max())

        # estimate step size by median diff (robust)
        d = np.diff(freq)
        steps.append(np.median(d[d > 0]))

    fmin = max(mins)   # intersection lower bound
    fmax = min(maxs)   # intersection upper bound
    step = min(steps)  # finest step observed

    if fmin >= fmax:
        raise ValueError(f"No overlapping frequency range across files: fmin={fmin}, fmax={fmax}")

    # Build grid: ensure inclusive end by a tiny epsilon
    freq_tgt = np.arange(fmin, fmax + step * 0.1, step, dtype=np.float32)

    return freq_tgt


def plot_txt_valid_spectra_by_day(file_stats_df):
    """
    Plot total valid spectra per organ by day for TXT dataset.
    For TXT data, 'valid spectra' means all spectra stored in each TXT file,
    since there is no inner/outer mask stage.
    """
    count_df = (
        file_stats_df.groupby(["day", "organ"])["n_valid_spectra"]
        .sum()
        .reset_index()
    )

    organs = sorted(file_stats_df["organ"].unique())
    day_values = sorted(file_stats_df["day"].unique())

    x = np.arange(len(organs))
    width = 0.25

    plt.figure(figsize=(12, 5))

    for i, day in enumerate(day_values):
        sub = (
            count_df[count_df["day"] == day]
            .set_index("organ")
            .reindex(organs, fill_value=0)
        )
        plt.bar(x + i * width, sub["n_valid_spectra"], width=width, label=f"Day {day}")

    plt.xticks(x + width, organs, rotation=45, ha="right")
    plt.ylabel("Total valid spectra")
    plt.title("Total valid spectra per organ by day (TXT dataset)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def build_from_metadata(meta_csv: str):
    meta = pd.read_csv(meta_csv)

    required = {"file_id", "path", "day", "organ", "condition"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {missing}")

    # 1) compute common grid
    freq_tgt = compute_common_grid(meta_csv)
    print(f"Common freq grid: {len(freq_tgt)} bands, range [{freq_tgt.min():.3f}, {freq_tgt.max():.3f}] THz")

    X_list, y_list, groups_list, days_list, cond_list = [], [], [], [], []

    # new: file-level stats for TXT dataset
    file_stats_rows = []

    for _, row in meta.iterrows():
        file_id = str(row["file_id"])
        rel_path = str(row["path"])
        day = int(row["day"])
        organ = str(row["organ"]).strip().lower()
        condition = str(row["condition"]).strip().lower()

        full_path = DATA_ROOT / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")

        freq_src, spectra_src = load_txt_spectra(str(full_path))  # (N, B_src)

        # TXT dataset has no inner/outer mask
        n_valid_spectra = spectra_src.shape[0]

        file_stats_rows.append({
            "file_id": file_id,
            "organ": organ,
            "day": day,
            "condition": condition,
            "n_valid_spectra": int(n_valid_spectra),
        })

        spectra = interp_spectra(freq_src, spectra_src, freq_tgt)  # (N, B_tgt)

        for s in spectra:
            X_list.append(s)
            y_list.append(organ)
            groups_list.append(file_id)
            days_list.append(day)
            cond_list.append(condition)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    groups = np.array(groups_list)
    days = np.array(days_list)
    cond = np.array(cond_list)

    file_stats_df = pd.DataFrame(file_stats_rows)

    return freq_tgt, X, y, groups, days, cond, file_stats_df


if __name__ == "__main__":
    freq, X, y, groups, days, cond, file_stats_df = build_from_metadata("metadata/metadata.csv")

    print("Built dataset:")
    print("Freq bands:", freq.shape)
    print("X:", X.shape, "y:", y.shape)
    print("Unique organs:", np.unique(y))
    print("Unique days:", np.unique(days))

    Path("artifacts").mkdir(exist_ok=True)
    np.save("artifacts/freq_thz.npy", freq)
    np.save("artifacts/X.npy", X)
    np.save("artifacts/y.npy", y)
    np.save("artifacts/groups.npy", groups)
    np.save("artifacts/days.npy", days)
    np.save("artifacts/cond.npy", cond)

    # save TXT file-level stats
    file_stats_df.to_csv("artifacts/txt_file_stats.csv", index=False)

    print("\nTXT file-level valid spectra stats:")
    print(file_stats_df)

    print("Saved to artifacts/*.npy")
    print("Saved TXT stats to artifacts/txt_file_stats.csv")

    # plot
    plot_txt_valid_spectra_by_day(file_stats_df)