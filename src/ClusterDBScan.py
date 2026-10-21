"""Part 2 -- discover classes in REAL sensor data with DBSCAN, then score
the discovered clusters against the true lighting labels.

DBSCAN sees only the two filtered sensor columns, never the labels.
The labels are used afterwards, only to build the confusion table.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from PipelineCommon import load_split_scaled, VIEW_LIM

EPS = 0.08           # <-- set from the sweep below
MIN_SAMPLES = 8
SWEEP = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12]

os.makedirs("/figures/generated", exist_ok=True)

data = load_split_scaled()
X = data["train"]          # clustering sees TRAINING data only
y_true = data["y_train"]   # used ONLY for scoring, never for clustering

# --- eps sweep: record this table in the paper ---
print("eps sweep (training set):")
for eps in SWEEP:
    lab = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(X)
    n = len(set(lab) - {-1})
    print(f"  eps={eps:.2f}: {n} clusters, {(lab == -1).sum()} noise")

# --- final clustering at the chosen eps ---
labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(X)
classes = sorted(set(labels) - {-1})
print(f"\neps={EPS}: {len(classes)} clusters; {(labels == -1).sum()} noise")
for c in classes:
    print(f"  cluster {c}: {(labels == c).sum()} points")
np.savetxt("labels.csv", labels, fmt="%d")

# --- confusion table: true label x discovered cluster ---
confusion = pd.crosstab(pd.Series(y_true, name="true label"),
                        pd.Series(labels, name="DBSCAN cluster"))
print("\nconfusion table (rows = true label, cols = DBSCAN cluster, -1 = noise):")
print(confusion)
confusion.to_csv("confusion_dbscan_vs_truth.csv")

# Suggested cluster -> name mapping (majority true label per cluster).
# Check it by eye before using it for CLASS_NAMES in model.h.
print("\nmajority label per cluster:")
for c in classes:
    vals, counts = np.unique(y_true[labels == c], return_counts=True)
    i = counts.argmax()
    print(f"  cluster {c} -> {vals[i]} ({counts[i]}/{counts.sum()} = "
          f"{100 * counts[i] / counts.sum():.1f}%)")

# --- k-means for contrast (only if DBSCAN found something) ---
kmeans_labels = None
if len(classes) > 0:
    kmeans_labels = KMeans(n_clusters=len(classes), n_init=10,
                           random_state=0).fit_predict(X)

# --- figure ---
MARKERS = ["o", "s", "^", "D", "v", "P"]
GRAYS = ["0.75", "0.45", "0.15", "0.6", "0.3", "0.85"]


def draw(ax, lab, title, show_noise):
    for i, c in enumerate(sorted(set(lab) - {-1}, key=str)):
        pts = X[lab == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=5,
                   marker=MARKERS[i % len(MARKERS)],
                   facecolor=GRAYS[i % len(GRAYS)], edgecolor="black",
                   linewidth=0.25, label=f"{c} (n={len(pts)})")
    if show_noise and (lab == -1).any():
        noise = X[lab == -1]
        ax.scatter(noise[:, 0], noise[:, 1], s=9, c="0.45", marker="x",
                   linewidth=0.5, label=f"noise, $-1$ (n={len(noise)})")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("scaled UV (filtered)")
    ax.set_aspect("equal")
    ax.set_xlim(*VIEW_LIM)
    ax.set_ylim(*VIEW_LIM)
    ax.legend(fontsize=6, loc="best")


panels = [(labels, f"DBSCAN (eps={EPS})", True),
          (y_true, "true labels", False)]
if kmeans_labels is not None:
    panels.append((kmeans_labels, "k-means (contrast)", False))

fig, axes = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), 3.4),
                         sharex=True, sharey=True)
for ax, (lab, title, noise) in zip(np.atleast_1d(axes), panels):
    draw(ax, np.asarray(lab), title, noise)
np.atleast_1d(axes)[0].set_ylabel("scaled IR (filtered)")
fig.savefig("/figures/generated/clustered_space.pdf",
            bbox_inches="tight", transparent=True)
plt.show()