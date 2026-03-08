import numpy as np
import matplotlib.pyplot as plt

# load dataset you already built
X = np.load("artifacts/X.npy")          # spectra (N, B)
y = np.load("artifacts/y.npy", allow_pickle=True)
freq = np.load("artifacts/freq_thz.npy")  # frequency axis

organs = np.unique(y)

plt.figure(figsize=(10,6))

for organ in organs:
    
    mask = y == organ
    spectra = X[mask]

    mean_spectrum = spectra.mean(axis=0)

    plt.plot(freq, mean_spectrum, label=organ)

plt.xlabel("Frequency (THz)")
plt.ylabel("Intensity (a.u.)")
plt.title("Mean Spectrum per Organ")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()