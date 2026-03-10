import numpy as np
import pandas as pd

X = np.load("artifacts/X_h5.npy")
y = np.load("artifacts/y_h5.npy")
groups = np.load("artifacts/groups_h5.npy", allow_pickle=True)
days = np.load("artifacts/days_h5.npy")
cond = np.load("artifacts/cond_h5.npy", allow_pickle=True)

mapping = pd.read_csv("artifacts/organ_mapping_h5.csv")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("groups:", len(np.unique(groups)))
print("days:", np.unique(days))
print("cond:", np.unique(cond))

print("\nSamples per organ label:")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    organ = mapping.loc[mapping["label"] == u, "organ"].values[0]
    print(f"{organ}: {c}")

print("\nSamples per day:")
for d in np.unique(days):
    print(f"day {d}: {(days == d).sum()}")

print("\nSamples per condition:")
for c in np.unique(cond):
    print(f"{c}: {(cond == c).sum()}")