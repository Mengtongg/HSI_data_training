import numpy as np
import matplotlib.pyplot as plt

# load dataset
X = np.load("artifacts/X.npy")
y = np.load("artifacts/y.npy", allow_pickle=True)
freq = np.load("artifacts/freq_thz.npy")
days = np.load("artifacts/days.npy")   

organs = np.unique(y)
unique_days = np.unique(days)

# create subplots
fig, axes = plt.subplots(1, len(unique_days), figsize=(15, 5), sharey=True)

for i, day in enumerate(unique_days):

    ax = axes[i]
    
    for organ in organs:
        
        mask = (y == organ) & (days == day)
        spectra = X[mask]

        if len(spectra) == 0:
            continue

        mean_spectrum = spectra.mean(axis=0)

        ax.plot(freq, mean_spectrum, label=organ)

    ax.set_title(f"Day {day}")
    ax.set_xlabel("Frequency (THz)")
    ax.grid(True)


axes[0].set_ylabel("Mean Intensity (a.u.)")


axes[-1].legend(loc="upper right", fontsize=8)

plt.suptitle("Mean Spectrum per Organ by Acquisition Day")
plt.tight_layout()
plt.show()