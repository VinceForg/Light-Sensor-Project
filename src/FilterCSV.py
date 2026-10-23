"""Filter raw sensor logs with an EMA, per class, in logging order.

Input : combined.csv  (no header: sensor1, sensor2, label) -- raw readings
Output: combined_filtered.csv  -- training data (test rows removed)
        heldout_filtered.csv   -- the 'test' rows, kept OUT of training
        ../figures/generated/filter_raw_vs_filtered.pdf
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ALPHA = 0.2               # EMA weight: y_n = a*x_n + (1-a)*y_{n-1}
RAW_CSV = "../figures/generated/reading.csv"
TEST_LABEL = "test"
PLOT_CLASS = None         # e.g. "led"; None = first class in the file


def ema(x, alpha=ALPHA):
    y = np.empty(len(x))
    y[0] = x[0]
    for n in range(1, len(x)):
        y[n] = alpha * x[n] + (1 - alpha) * y[n - 1]
    return y


def filter_per_class(df):
    """EMA each class separately, in its original order, so the filter
    never blends the end of one class into the start of the next."""
    df = df.copy()
    for col in ("sensor1", "sensor2"):
        df[col + "_filt"] = df.groupby("label", sort=False)[col].transform(
            lambda s: ema(s.to_numpy(dtype=float)))
    return df


raw = pd.read_csv(RAW_CSV, header=None, names=["sensor1", "sensor2", "label"])
raw["label"] = raw["label"].astype(str).str.strip()

# Keep held-out test rows OUT of the training data.
train_raw = raw[raw["label"] != TEST_LABEL]
test_raw = raw[raw["label"] == TEST_LABEL]

train = filter_per_class(train_raw)
train.to_csv("combined_filtered.csv", index=False)
print(f"training rows: {len(train)}  -> combined_filtered.csv")

if len(test_raw):
    test = filter_per_class(test_raw)
    test.to_csv("heldout_filtered.csv", index=False)
    print(f"held-out rows: {len(test)}  -> heldout_filtered.csv (NOT for training)")

# Rows per class (assignment requires >= 500 each).
print("\nrows per class:")
print(train["label"].value_counts().to_string())

# Per-class statistics: are the classes actually separable?
print("\nper-class stats (filtered):")
print(train.groupby("label")[["sensor1_filt", "sensor2_filt"]]
           .agg(["mean", "std", "min", "max"]).round(1).to_string())

# Part 5 evidence plot: raw vs filtered for one class.
cls = PLOT_CLASS or train["label"].iloc[0]
sub = train[train["label"] == cls].reset_index(drop=True)
fig, axes = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
for ax, col, name in zip(axes, ("sensor1", "sensor2"), ("UV", "IR")):
    ax.plot(sub[col], lw=0.6, color="0.6", label="raw")
    ax.plot(sub[col + "_filt"], lw=1.2, color="black", label="filtered")
    ax.set_ylabel(f"{name} (ADC)")
    ax.legend(fontsize=8)
axes[-1].set_xlabel("sample index")
axes[0].set_title(f"Raw vs EMA-filtered (alpha={ALPHA}), class '{cls}'")
fig.savefig("/figures/generated/filter_raw_vs_filtered.pdf",
            bbox_inches="tight")
plt.show()