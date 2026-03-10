import h5py
from pathlib import Path

h5_path = Path(r"D:\Hyperspectral_Data\All Data\Day 1\Brain 10 ms\251007_193416_hyper_brain_10ms_hyp_SpectralHypercube.h5")

with h5py.File(h5_path, "r") as f:
    print("Top-level keys:")
    for key in f.keys():
        print("-", key)

    group = f["SpectralHypercube"]

    print("\nType of SpectralHypercube:", type(group))
    print("\nKeys inside SpectralHypercube:")

    for subkey in group.keys():
        obj = group[subkey]
        print("-", subkey)
        if isinstance(obj, h5py.Dataset):
            print("  shape:", obj.shape)
            print("  dtype:", obj.dtype)
        elif isinstance(obj, h5py.Group):
            print("  type: Group")