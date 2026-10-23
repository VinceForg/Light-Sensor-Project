"""Part 4 -- on-device vs desktop agreement, and accuracy against true labels.

Input: heldout_device.csv -- Serial Monitor output of LightClassifier.ino:
    label,uv_raw,ir_raw,uv_filt,ir_filt,prediction,winning_value
Lines starting with '#' (benchmark output) are ignored.

For every row, the desktop recomputes the prediction from the SAME filtered
values the device used (uv_filt, ir_filt), with the same log transform,
bounds, tables and threshold, and compares it to the device's prediction.
"""
import numpy as np
import pandas as pd
from PipelineCommon import (load_split_scaled, scale_features,
                            cell_of, quantize, dequantize, REJECT_TAU)

HELDOUT = "figures/generated/heldout.csv"     # edit if you saved it elsewhere
SURFACES = "figures/generated/surfaces.npz"          # same file GenerateModelHeader.py read

# Must match CLUSTER_NAMES in GenerateModelHeader.py.
CLUSTER_NAMES = {0: "shade", 1: "sun", 2: "indoor", 3: "shade"}

# What each true lighting condition SHOULD be classified as.
EXPECTED = {"incandescent": "indoor", "led": "indoor",
            "shade": "shade", "sun": "sun"}

# --- the model, exactly as shipped ---------------------------------------
surfaces = np.load(SURFACES)
classes = sorted(int(k[1:]) for k in surfaces.files)
tables = {c: quantize(surfaces[f"c{c}"]) for c in classes}
bounds = load_split_scaled()["bounds"]


def desktop_predict(uv_filt, ir_filt):
    x = np.log1p(np.array([uv_filt, ir_filt]) * 4095)     # same as firmware
    p, _ = scale_features(x, bounds)
    r, c = cell_of(p)
    vals = np.array([dequantize(tables[k][r, c]) for k in classes])
    best = int(np.argmax(vals))
    return "UNKNOWN" if vals[best] < REJECT_TAU else CLUSTER_NAMES[classes[best]]


# --- load the device log ---------------------------------------------------
df = pd.read_csv(HELDOUT, comment="#", skipinitialspace=True)
df = df[df["label"] != "label"]                            # drop repeated headers
df["label"] = df["label"].str.strip().str.lower()
df = df[df["label"] != "none"]                             # untagged rows
for col in ("uv_filt", "ir_filt"):
    df[col] = df[col].astype(float)
df["prediction"] = df["prediction"].str.strip()

df["desktop"] = [desktop_predict(u, i) for u, i in zip(df["uv_filt"], df["ir_filt"])]
df["expected"] = df["label"].map(EXPECTED)

# --- results ---------------------------------------------------------------
print(f"held-out rows: {len(df)}")
print(df["label"].value_counts().to_string(), "\n")

agree = (df["prediction"] == df["desktop"]).mean()
print(f"device vs desktop agreement: {100 * agree:.1f}%")
if agree < 1:
    print("disagreeing rows:")
    print(df.loc[df["prediction"] != df["desktop"],
                 ["label", "uv_filt", "ir_filt", "prediction", "desktop"]].to_string())

acc = (df["prediction"] == df["expected"]).mean()
print(f"\ndevice accuracy vs true labels: {100 * acc:.1f}%")
print("\nconfusion (rows = true condition, cols = device prediction):")
print(pd.crosstab(df["label"], df["prediction"]).to_string())

print("\nper-condition accuracy:")
for lab, g in df.groupby("label"):
    print(f"  {lab:>12}: {100 * (g['prediction'] == g['expected']).mean():5.1f}%  "
          f"(expected '{EXPECTED.get(lab)}', n={len(g)})")
