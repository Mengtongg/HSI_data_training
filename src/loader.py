import numpy as np
import pandas as pd


def load_txt_spectra(txt_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one txt file exported with columns:
        Frequency [THz] | Spectrum_Ave [a.u.] (repeated) | Spectrum_Std [a.u.] (repeated)

    Returns:
        freq_thz: shape (B,)
        spectra:  shape (N, B) where N = number of selected points (Spectrum_Ave columns)
    """
    # Tab-separated with a header row
    df = pd.read_csv(txt_path, sep="\t", engine="python")

    # First column should be frequency
    freq_col = df.columns[0]
    freq_thz = df[freq_col].to_numpy(dtype=np.float32)

    # Select ONLY Spectrum_Ave columns (ignore Spectrum_Std for baseline)
    ave_cols = [c for c in df.columns if "Spectrum_Ave" in c]
    if len(ave_cols) == 0:
        raise ValueError(f"No Spectrum_Ave columns found in {txt_path}. Columns: {list(df.columns)}")

    # df[ave_cols] is (B, N) -> transpose to (N, B)
    spectra = df[ave_cols].to_numpy(dtype=np.float32).T

    return freq_thz, spectra