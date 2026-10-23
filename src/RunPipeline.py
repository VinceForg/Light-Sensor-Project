"""Run the whole offline pipeline in order, stopping at the first failure.
 
Put this file in src/ next to the other scripts and run:
    python RunPipeline.py
 
Edit STEPS below if your file names differ. Steps whose file does not
exist yet (e.g. GenerateModelHeader.py) are skipped with a warning.
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path
 
SRC_DIR = Path(__file__).resolve().parent
 
# (script file, what it produces) -- in the order they must run.
STEPS = [
    ("FilterCSV.py",          "combined_filtered.csv + raw-vs-filtered plot"),
    ("ClusterDBScan.py",       "labels.csv + confusion table + clustered_space.pdf"),
    ("BandwidthSweep.py",      "validation-selected bandwidth h"),
    ("KdeClassContours.py",    "surfaces.npz + continuous class maps"),
    ("KdeTotalSurface.py",     "3D decision surface"),
    ("DiscretizeSurfaces.py",  "footprint table + uint8 error + stored maps"),
    ("EvaluateTest.py",        "stored-vs-continuous agreement on test split"),
    ("GenerateModelHeader.py", "model.h"),
]
 
# Figures are saved, not shown, so plt.show() never blocks the run.
env = dict(os.environ, MPLBACKEND="Agg")
 
# The bandwidth the KDE scripts actually use (if defined in PipelineCommon).
sys.path.insert(0, str(SRC_DIR))
try:
    from PipelineCommon import BANDWIDTH as CONFIGURED_H
except ImportError:
    CONFIGURED_H = None
 
log_path = SRC_DIR / "../figures/generated/pipeline_log.txt"
log = open(log_path, "w", encoding="utf-8")
 
 
def out(text=""):
    print(text)
    log.write(text + "\n")
 
 
start = time.time()
for i, (script, produces) in enumerate(STEPS, 1):
    path = SRC_DIR / script
    out(f"\n{'=' * 70}\n[{i}/{len(STEPS)}] {script}  ->  {produces}\n{'=' * 70}")
 
    if not path.exists():
        out(f"  SKIPPED: {script} not found in {SRC_DIR}")
        continue
 
    result = subprocess.run([sys.executable, str(path)], cwd=SRC_DIR, env=env,
                            capture_output=True, text=True)
    if result.stdout:
        out(result.stdout.rstrip())
 
    if result.returncode != 0:
        out(result.stderr.rstrip())
        out(f"\nFAILED at step {i}: {script}. Fix it and re-run.")
        log.close()
        sys.exit(1)
 
    # Guard: warn if the sweep picked a different h than the KDE scripts use.
    if script == "BandwidthSweep.py":
        m = re.search(r"selects h = ([\d.]+)", result.stdout)
        if m and CONFIGURED_H is not None and abs(float(m.group(1)) - CONFIGURED_H) > 1e-9:
            out(f"\n  WARNING: sweep selected h = {m.group(1)}, but PipelineCommon.BANDWIDTH "
                f"= {CONFIGURED_H}.\n  Update BANDWIDTH and re-run, or justify the "
                f"difference in the paper.")
        sweep_vals = re.findall(r"^\s*([\d.]+)\s+-?[\d.]+\s*$", result.stdout, re.M)
        if m and sweep_vals and m.group(1) in (sweep_vals[0], sweep_vals[-1]):
            out("  WARNING: selected h is at the edge of SWEEP -- extend the range.")
 
out(f"\nDone in {time.time() - start:.1f} s. Full output saved to {log_path.name}")
log.close()