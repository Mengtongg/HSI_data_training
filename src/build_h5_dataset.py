from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import os
import matplotlib.pyplot as plt

from configs.paths import DATA_ROOT


# Parameters
META_CSV = "metadata/metadata_h5.csv"

OUTPUT_ROOT = Path("artifacts") / "training_h5_2"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

THRESHOLD_PERCENTILE = int(os.environ.get("HSI_THRESH", 85))
INNER_PERCENTILE = int(os.environ.get("HSI_INNER", 50))
PIXELS_PER_FILE = int(os.environ.get("HSI_PIXELS", 1000))
RANDOM_SEED = 42


def load_single_h5(h5_path: Path):
    """
    Read a single h5 file.

    returns:
        cube: (B, H, W)
        freq: (B,)
        sat:  (H, W)
    """
    with h5py.File(h5_path, "r") as f:
        cube = f["SpectralHypercube"]["Hyperspectrum_cube"][:]   # (300,848,848)
        freq = f["SpectralHypercube"]["fr_real"][:].squeeze()    # (300,)
        sat = f["SpectralHypercube"]["saturationMap"][:]         # (848,848)

    cube = np.asarray(cube, dtype=np.float32)
    freq = np.asarray(freq, dtype=np.float32)
    sat = np.asarray(sat)

    return cube, freq, sat


def compute_common_freq_grid(meta_csv: str):
    """
    Build a common overlapping frequency grid across all H5 files.
    """
    meta = pd.read_csv(meta_csv)

    mins = []
    maxs = []
    steps = []

    for _, row in meta.iterrows():
        rel_path = str(row["path"]).strip()
        full_path = DATA_ROOT / rel_path

        with h5py.File(full_path, "r") as f:
            freq = f["SpectralHypercube"]["fr_real"][:].squeeze()

        freq = np.asarray(freq, dtype=np.float32)

        mins.append(freq.min())
        maxs.append(freq.max())

        d = np.diff(freq)
        d = d[d > 0]
        if len(d) == 0:
            raise ValueError(f"Invalid frequency axis in {full_path}")
        steps.append(np.median(d))

    fmin = max(mins)
    fmax = min(maxs)
    step = min(steps)

    if fmin >= fmax:
        raise ValueError(f"No overlapping frequency range found: fmin={fmin}, fmax={fmax}")

    freq_tgt = np.arange(fmin, fmax + 0.1 * step, step, dtype=np.float32)
    return freq_tgt


def interp_spectra(freq_src, spectra_src, freq_tgt):
    """
    freq_src: (B_src,)
    spectra_src: (N, B_src)
    freq_tgt: (B_tgt,)
    returns: (N, B_tgt)
    """
    out = np.empty((spectra_src.shape[0], len(freq_tgt)), dtype=np.float32)
    for i in range(spectra_src.shape[0]):
        out[i] = np.interp(freq_tgt, freq_src, spectra_src[i])
    return out


def extract_valid_spectra(
    cube,
    sat,
    threshold_percentile=85,
    inner_percentile=50,
    pixels_per_file=1000,
    rng=None
):
    """
    Extract valid spectra from a single h5 cube.

    logic:
      1. mean intensity image
      2. outer mask = mean_img > outer percentile
      3. sat mask = sat == 1
      4. valid outer mask = outer mask & sat mask
      5. inner threshold is computed inside valid outer mask
      6. inner mask = valid outer mask & (mean_img > inner threshold)
      7. randomly sample fixed number of valid pixels

    returns:
      spectra: (N, B)
      inner_mask: (H, W)
      stats: dict with valid pixel counts before sampling
    """
    if rng is None:
        rng = np.random.default_rng()

    mean_img = cube.mean(axis=0)  # (H, W)

    # ===== outer mask =====
    outer_thresh = np.percentile(mean_img, threshold_percentile)
    outer_mask = mean_img > outer_thresh

    # sat == 1 is valid
    sat_mask = (sat == 1)

    valid_outer_mask = outer_mask & sat_mask
    n_valid_outer = int(valid_outer_mask.sum())

    if n_valid_outer == 0:
        stats = {
            "n_valid_outer": 0,
            "n_valid_inner": 0,
            "n_sampled": 0,
        }
        return np.empty((0, cube.shape[0]), dtype=np.float32), valid_outer_mask, stats

    # ===== inner mask =====
    inner_values = mean_img[valid_outer_mask]
    inner_thresh = np.percentile(inner_values, inner_percentile)
    inner_mask = valid_outer_mask & (mean_img > inner_thresh)

    valid_pixels = np.where(inner_mask)
    n_valid_inner = len(valid_pixels[0])

    if n_valid_inner == 0:
        stats = {
            "n_valid_outer": n_valid_outer,
            "n_valid_inner": 0,
            "n_sampled": 0,
        }
        return np.empty((0, cube.shape[0]), dtype=np.float32), inner_mask, stats

    sample_size = min(pixels_per_file, n_valid_inner)
    chosen_idx = rng.choice(n_valid_inner, size=sample_size, replace=False)

    xs = valid_pixels[0][chosen_idx]
    ys = valid_pixels[1][chosen_idx]

    # cube shape = (B, H, W)
    # selected spectra from cube, converted to (N, B)
    spectra = cube[:, xs, ys].T.astype(np.float32)

    stats = {
        "n_valid_outer": n_valid_outer,
        "n_valid_inner": int(n_valid_inner),
        "n_sampled": int(sample_size),
    }

    return spectra, inner_mask, stats


def plot_valid_spectra_by_day(stats_df, value_col="n_valid_inner"):
    """
    Plot total valid spectra per organ by day before sampling.
    value_col can be:
      - n_valid_outer
      - n_valid_inner
      - n_sampled
    """
    count_df = (
        stats_df.groupby(["day", "organ"])[value_col]
        .sum()
        .reset_index()
    )

    organs = sorted(stats_df["organ"].unique())
    day_values = sorted(stats_df["day"].unique())

    x = np.arange(len(organs))
    width = 0.25

    plt.figure(figsize=(12, 5))

    for i, day in enumerate(day_values):
        sub = (
            count_df[count_df["day"] == day]
            .set_index("organ")
            .reindex(organs, fill_value=0)
        )
        plt.bar(x + i * width, sub[value_col], width=width, label=f"Day {day}")

    plt.xticks(x + width, organs, rotation=45, ha="right")

    ylabel_map = {
        "n_valid_outer": "Total valid spectra before inner mask",
        "n_valid_inner": "Total valid spectra before sampling",
        "n_sampled": "Total sampled spectra",
    }
    title_map = {
        "n_valid_outer": "Total valid spectra per organ by day (outer mask)",
        "n_valid_inner": "Total valid spectra per organ by day (before sampling)",
        "n_sampled": "Total sampled spectra per organ by day",
    }

    plt.ylabel(ylabel_map.get(value_col, value_col))
    plt.title(title_map.get(value_col, value_col))
    plt.legend()
    plt.tight_layout()
    plt.show()


def build_h5_dataset(meta_csv: str):
    meta = pd.read_csv(meta_csv)

    required = {"file_id", "path", "day", "organ", "condition"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {missing}")

    # organ to label mapping
    organ_names = sorted(meta["organ"].astype(str).str.strip().str.lower().unique())
    organ_to_label = {name: i for i, name in enumerate(organ_names)}

    print("Organ mapping:")
    for k, v in organ_to_label.items():
        print(f"  {k}: {v}")

    print("\nComputing common frequency grid...")
    freq_tgt = compute_common_freq_grid(meta_csv)
    print(f"Common freq grid: {len(freq_tgt)} bands")
    print(f"Range: [{freq_tgt.min():.3f}, {freq_tgt.max():.3f}]")

    X_list = []
    y_list = []
    groups_list = []
    days_list = []
    cond_list = []

    # new: file-level stats before sampling
    file_stats_rows = []

    rng = np.random.default_rng(RANDOM_SEED)

    for _, row in meta.iterrows():
        file_id = str(row["file_id"]).strip()
        rel_path = str(row["path"]).strip()
        day = int(row["day"])
        organ = str(row["organ"]).strip().lower()
        condition = str(row["condition"]).strip().lower()

        full_path = DATA_ROOT / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")

        print(f"\nProcessing: {file_id}")
        cube, freq_src, sat = load_single_h5(full_path)

        spectra_src, valid_mask, stats = extract_valid_spectra(
            cube=cube,
            sat=sat,
            threshold_percentile=THRESHOLD_PERCENTILE,
            inner_percentile=INNER_PERCENTILE,
            pixels_per_file=PIXELS_PER_FILE,
            rng=rng
        )

        # save file-level stats no matter sampled or skipped
        file_stats_rows.append({
            "file_id": file_id,
            "organ": organ,
            "day": day,
            "condition": condition,
            "n_valid_outer": stats["n_valid_outer"],
            "n_valid_inner": stats["n_valid_inner"],
            "n_sampled": stats["n_sampled"],
        })

        print(f"  valid outer pixels : {stats['n_valid_outer']}")
        print(f"  valid inner pixels : {stats['n_valid_inner']}")
        print(f"  sampled spectra    : {stats['n_sampled']}")

        if spectra_src.shape[0] == 0:
            print("  WARNING: no valid spectra extracted, skipping.")
            continue

        spectra = interp_spectra(freq_src, spectra_src, freq_tgt)
        print("  extracted spectra:", spectra.shape)

        labels = np.full(spectra.shape[0], organ_to_label[organ], dtype=np.int64)
        groups = np.array([file_id] * spectra.shape[0], dtype=object)
        days = np.full(spectra.shape[0], day, dtype=np.int64)
        conds = np.array([condition] * spectra.shape[0], dtype=object)

        X_list.append(spectra)
        y_list.append(labels)
        groups_list.append(groups)
        days_list.append(days)
        cond_list.append(conds)

    if len(X_list) == 0:
        raise ValueError("No spectra extracted from any h5 file.")

    X = np.vstack(X_list)                      # (N, B)
    y = np.concatenate(y_list)                # (N,)
    groups = np.concatenate(groups_list)      # (N,)
    days = np.concatenate(days_list)          # (N,)
    cond = np.concatenate(cond_list)          # (N,)

    file_stats_df = pd.DataFrame(file_stats_rows)

    return freq_tgt, X, y, groups, days, cond, organ_to_label, file_stats_df


if __name__ == "__main__":
    freq, X, y, groups, days, cond, organ_to_label, file_stats_df = build_h5_dataset(META_CSV)

    suffix = f"t{THRESHOLD_PERCENTILE}_i{INNER_PERCENTILE}_p{PIXELS_PER_FILE}"

    print("\nFinal dataset:")
    print("  freq:", freq.shape)
    print("  X   :", X.shape)
    print("  y   :", y.shape)
    print("  groups unique:", len(np.unique(groups)))
    print("  days unique  :", np.unique(days))
    print("  cond unique  :", np.unique(cond))

    np.save(OUTPUT_ROOT / f"freq_h5_{suffix}.npy", freq)
    np.save(OUTPUT_ROOT / f"X_h5_{suffix}.npy", X)
    np.save(OUTPUT_ROOT / f"y_h5_{suffix}.npy", y)
    np.save(OUTPUT_ROOT / f"groups_h5_{suffix}.npy", groups)
    np.save(OUTPUT_ROOT / f"days_h5_{suffix}.npy", days)
    np.save(OUTPUT_ROOT / f"cond_h5_{suffix}.npy", cond)

    # save label mapping
    mapping_df = pd.DataFrame({
        "organ": list(organ_to_label.keys()),
        "label": list(organ_to_label.values())
    })
    mapping_df.to_csv(OUTPUT_ROOT / f"organ_mapping_h5_{suffix}.csv", index=False)

    # save file-level stats
    stats_csv = OUTPUT_ROOT / f"file_stats_h5_{suffix}.csv"
    file_stats_df.to_csv(stats_csv, index=False)

    print("\nFile-level valid pixel stats:")
    print(file_stats_df)

    print("\nSaved:")
    print(OUTPUT_ROOT)
    print(stats_csv)

    # plots
    plot_valid_spectra_by_day(file_stats_df, value_col="n_valid_inner")