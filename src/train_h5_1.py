import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


def majority_vote_per_file(y_pred, groups):
    votes = {}
    for pred, gid in zip(y_pred, groups):
        votes.setdefault(gid, []).append(pred)
    out = {}
    for gid, preds in votes.items():
        vals, counts = np.unique(preds, return_counts=True)
        out[gid] = vals[np.argmax(counts)]
    return out


def main():
    X = np.load("artifacts/X_h5.npy")
    y = np.load("artifacts/y_h5.npy")
    groups = np.load("artifacts/groups_h5.npy", allow_pickle=True)
    days = np.load("artifacts/days_h5.npy")
    mapping = pd.read_csv("artifacts/organ_mapping_h5.csv")

    label_to_organ = dict(zip(mapping["label"], mapping["organ"]))

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=4000))
    ])

    for test_day in [1, 2, 3]:
        train_mask = days != test_day
        test_mask = days == test_day

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        groups_test = groups[test_mask]

        # per-spectrum L2 normalize first
        X_train = normalize(X_train, norm="l2")
        X_test = normalize(X_test, norm="l2")

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        target_names = [label_to_organ[i] for i in sorted(np.unique(y_test))]
        labels_sorted = sorted(np.unique(y_test))

        print("\n==============================")
        print(f"TEST DAY = {test_day}")
        print(f"Spectrum-level: acc={acc:.3f}, macroF1={f1:.3f}")
        print(classification_report(
            y_test,
            y_pred,
            labels=labels_sorted,
            target_names=target_names,
            zero_division=0
        ))

        # file-level majority vote
        file_pred = majority_vote_per_file(y_pred, groups_test)
        file_true = {}
        for yt, gid in zip(y_test, groups_test):
            file_true.setdefault(gid, yt)

        file_ids = sorted(file_true.keys())
        y_file_true = np.array([file_true[fid] for fid in file_ids])
        y_file_pred = np.array([file_pred[fid] for fid in file_ids])

        file_acc = accuracy_score(y_file_true, y_file_pred)
        print(f"File-level: acc={file_acc:.3f} (n_files={len(file_ids)})")


if __name__ == "__main__":
    main()