import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

h5_path = Path(r"D:\Hyperspectral_Data\All Data\Day 1\Brain 10 ms\251007_193416_hyper_brain_10ms_hyp_SpectralHypercube.h5")

with h5py.File(h5_path, "r") as f:
    cube = f["SpectralHypercube"]["Hyperspectrum_cube"][:]   # shape (300, 848, 848)
    freq = f["SpectralHypercube"]["fr_real"][:].squeeze()    # shape (300,)
    sat = f["SpectralHypercube"]["saturationMap"][:]         # shape (848, 848)

print("cube shape:", cube.shape)
print("freq shape:", freq.shape)
print("sat shape:", sat.shape)
print("freq first 5:", freq[:5])
print("freq last 5:", freq[-5:])

# figure for 150th band
band_idx = 150
band_img = cube[band_idx]

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.imshow(band_img, cmap="gray")
plt.title(f"Band {band_idx}, freq={freq[band_idx]:.2f}")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(sat, cmap="hot")
plt.title("Saturation Map")
plt.colorbar()

# Mean intensity image
mean_img = cube.mean(axis=0)
plt.subplot(1, 3, 3)
plt.imshow(mean_img, cmap="gray")
plt.title("Mean Intensity Image")
plt.colorbar()

plt.tight_layout()
plt.show()