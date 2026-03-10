import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

h5_path = Path(r"D:\Hyperspectral_Data\All Data\Day 1\Brain 10 ms\251007_193416_hyper_brain_10ms_hyp_SpectralHypercube.h5")

with h5py.File(h5_path, "r") as f:
    cube = f["SpectralHypercube"]["Hyperspectrum_cube"][:]   # (300,848,848)
    freq = f["SpectralHypercube"]["fr_real"][:].squeeze()    # (300,)
    sat = f["SpectralHypercube"]["saturationMap"][:]         # (848,848)

print("cube shape:", cube.shape)

# 平均强度图
mean_img = cube.mean(axis=0)

# saturation mask
sat_mask = (sat == 1)

print("sat==1 pixels:", sat_mask.sum())

# 要测试的 thresholds
percentiles = [70, 80, 90]

plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.imshow(mean_img, cmap="gray")
plt.title("Mean intensity image")
plt.colorbar()

plt.subplot(1,3,2)
plt.hist(mean_img.ravel(), bins=100)
plt.title("Intensity histogram")

plt.subplot(1,3,3)
plt.imshow(sat_mask, cmap="gray")
plt.title("Saturation mask")

plt.tight_layout()
plt.show()

for p in percentiles:

    thresh = np.percentile(mean_img, p)

    tissue_mask = mean_img > thresh
    valid_mask = tissue_mask & sat_mask

    n_pixels = valid_mask.sum()

    print("\nPercentile:", p)
    print("threshold:", thresh)
    print("selected pixels:", n_pixels)

    # 画 mask
    plt.figure(figsize=(5,5))
    plt.imshow(valid_mask, cmap="gray")
    plt.title(f"Mask percentile {p}")
    plt.show()

    # 提 spectra
    spectra = cube[:, valid_mask].T

    if spectra.shape[0] == 0:
        continue

    # 随机抽 20 条画图
    n_plot = min(20, spectra.shape[0])
    idx = np.random.choice(spectra.shape[0], n_plot, replace=False)

    plt.figure(figsize=(8,5))

    for i in idx:
        plt.plot(freq, spectra[i], alpha=0.4)

    plt.xlabel("Frequency")
    plt.ylabel("Intensity")
    plt.title(f"Random spectra (percentile {p})")

    plt.tight_layout()
    plt.show()