import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

processed_file = Path(__file__).parents[1] / "output" / "ablation_study_results_no_gps.csv"
df = pd.read_csv(processed_file)

results = defaultdict(list)

skip_not = 0
skip_rectification = 0
skip_rectification_by_rule = defaultdict(int)
skip_no_mfr = 0
skip_nan = 0
for i, row_mpr in df[df["mpr"]].iterrows():
    if isinstance(row_mpr["robustness"], float):
        rob_mpr = [row_mpr["robustness"]]
    else:
        rob_mpr = list(map(float, row_mpr["robustness"].split(",")))

    if any([np.isnan(x) for x in rob_mpr]):
        skip_nan += 1
        continue

    if any([x in [-1e-3, 1e-3] for x in rob_mpr]):
        skip_rectification += 1
        skip_rectification_by_rule[row_mpr["rule"]] += 1
        continue

    row_mfr = df[
        (df["scenario_id"] == row_mpr["scenario_id"])
        & (df["rule"] == row_mpr["rule"])
        & (~df["mpr"])
        & (df["vehicle_id"] == row_mpr["vehicle_id"])
    ]

    if len(row_mfr) < 1:
        # no corresponding mfr found
        skip_no_mfr += 1
        continue

    assert len(row_mfr) == 1
    row_mfr = row_mfr.iloc[0]

    if isinstance(row_mfr["robustness"], float):
        rob_mfr = [row_mfr["robustness"]]
    else:
        rob_mfr = list(map(float, row_mfr["robustness"].split(",")))

    assert len(rob_mfr) == len(rob_mpr)

    results[f"{row_mpr['rule']}_violated"] += [any([x < 0 for x in rob_mfr])]

    skip_not += 1
    results[f"{row_mfr['rule']}_mfr"] += rob_mfr
    results[f"{row_mpr['rule']}_mpr"] += rob_mpr

for rule in ["R_G1", "R_G2", "R_G3", "R_G4", "R_I1", "R_I2", "R_I3", "R_I4", "R_I5"]:
    for typ in ["mfr", "mpr"]:
        values = results[f"{rule}_{typ}"]
        if len(values) < 1:
            continue
        results[f"{rule}_{typ}_agg"] = [np.ptp(values), np.std(values)]

print("hey")
