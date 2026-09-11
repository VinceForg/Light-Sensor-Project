"""Part 0 -- write the dataset to disk so every later stage reads the
same file, and so the file itself can be committed as evidence.

Run once: python generate_dataset.py"""
import numpy as np
from PipelineCommon import load_dataset

X = load_dataset() #synthetic by default
np.savetxt("figures/generated/dataset.csv", X, delimiter=",", fmt="%.6f")
print(f"wrote dataset.csv: {X.shape[0]} points, {X.shape[1]} features")