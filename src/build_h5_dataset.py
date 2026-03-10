from pathlib import Path
import numpy as np
import pandas as pd
import h5py

from configs.paths import DATA_ROOT


# Parameters 
META_CSV = "metadata/metadata_h5.csv"
OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)

THRESHOLD_PERCENTILE = 80
PIXELS_PER_FILE = 2000
RANDOM_SEED = 42


def load_single_h5(h5_path: Path):
    """
    Read a single h5 file
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


def extract_valid_spectra(cube, sat, threshold_percentile=80, pixels_per_file=2000, rng=None):
    """
    Extract valid spectra from a single h5 cube
    logic:
      1. mean intensity image
      2. tissue mask = mean_img > percentile
      3. sat mask = sat == 1
      4. valid mask = tissue & sat
      5. random sample fixed number of pixels

    returns:
      spectra: (N, B)
      valid_mask: (H, W)
    """
    if rng is None:
        rng = np.random.default_rng()

    mean_img = cube.mean(axis=0)  # (H,W)

    # foreground mask
    thresh = np.percentile(mean_img, threshold_percentile)
    tissue_mask = mean_img > thresh

    # sat==1 is normal
    sat_mask = (sat == 1)

    valid_mask = tissue_mask & sat_mask

    valid_pixels = np.where(valid_mask)
    n_valid = len(valid_pixels[0])

    if n_valid == 0:
        return np.empty((0, cube.shape[0]), dtype=np.float32), valid_mask

    sample_size = min(pixels_per_file, n_valid)
    chosen_idx = rng.choice(n_valid, size=sample_size, replace=False)

    xs = valid_pixels[0][chosen_idx]
    ys = valid_pixels[1][chosen_idx]

    # cube shape = (B,H,W)
    # selected spectra from cube, converted to (N,B)
    spectra = cube[:, xs, ys].T.astype(np.float32)

    return spectra, valid_mask


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

        print("  cube shape:", cube.shape)
        print("  freq shape:", freq_src.shape)
        print("  sat shape :", sat.shape)


        spectra_src, valid_mask = extract_valid_spectra(
            cube=cube,
            sat=sat,
            threshold_percentile=THRESHOLD_PERCENTILE,
            pixels_per_file=PIXELS_PER_FILE,
            rng=rng
        )


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

    X = np.vstack(X_list)                      # (N,B)
    y = np.concatenate(y_list)                # (N,)
    groups = np.concatenate(groups_list)      # (N,)
    days = np.concatenate(days_list)          # (N,)
    cond = np.concatenate(cond_list)          # (N,)

    return freq_tgt, X, y, groups, days, cond, organ_to_label


if __name__ == "__main__":
    freq, X, y, groups, days, cond, organ_to_label = build_h5_dataset(META_CSV)

    print("\nFinal dataset:")
    print("  freq:", freq.shape)
    print("  X   :", X.shape)
    print("  y   :", y.shape)
    print("  groups unique:", len(np.unique(groups)))
    print("  days unique  :", np.unique(days))
    print("  cond unique  :", np.unique(cond))

    np.save(OUTPUT_DIR / "freq_h5.npy", freq)
    np.save(OUTPUT_DIR / "X_h5.npy", X)
    np.save(OUTPUT_DIR / "y_h5.npy", y)
    np.save(OUTPUT_DIR / "groups_h5.npy", groups)
    np.save(OUTPUT_DIR / "days_h5.npy", days)
    np.save(OUTPUT_DIR / "cond_h5.npy", cond)

    # save label mapping
    mapping_df = pd.DataFrame({
        "organ": list(organ_to_label.keys()),
        "label": list(organ_to_label.values())
    })
    mapping_df.to_csv(OUTPUT_DIR / "organ_mapping_h5.csv", index=False)

    print("\nSaved:")
    print("  artifacts/freq_h5.npy")
    print("  artifacts/X_h5.npy")
    print("  artifacts/y_h5.npy")
    print("  artifacts/groups_h5.npy")
    print("  artifacts/days_h5.npy")
    print("  artifacts/cond_h5.npy")
    print("  artifacts/organ_mapping_h5.csv")