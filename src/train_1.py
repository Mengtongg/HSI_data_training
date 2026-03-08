import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


def majority_vote_per_file(y_pred, groups):
    """Return dict file_id -> predicted label by majority vote."""
    votes = {}
    for pred, gid in zip(y_pred, groups):
        votes.setdefault(gid, []).append(pred)
    out = {}
    for gid, preds in votes.items():
        vals, counts = np.unique(preds, return_counts=True)
        out[gid] = vals[np.argmax(counts)]
    return out


def main():
    X = np.load("artifacts/X.npy")
    y = np.load("artifacts/y.npy", allow_pickle=True)
    groups = np.load("artifacts/groups.npy", allow_pickle=True)
    days = np.load("artifacts/days.npy")

    # Pipeline: standardize + multinomial logistic regression
    clf = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=30)),
    ("lr", LogisticRegression(max_iter=4000))
    ])
    
    for test_day in [1, 2, 3]:
        train_mask = days != test_day
        test_mask = days == test_day

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        groups_test = groups[test_mask]

        # L2-normalize each spectrum before standardization (robust to intensity scale)
        X_train = normalize(X_train, norm="l2")
        X_test = normalize(X_test, norm="l2")

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        print("\n==============================")
        print(f"TEST DAY = {test_day}")
        print(f"Spectrum-level: acc={acc:.3f}, macroF1={f1:.3f}")
        print(classification_report(y_test, y_pred))

        # File-level vote
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