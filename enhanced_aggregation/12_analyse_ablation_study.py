from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

processed_file = Path(__file__).parents[1] / "output" / "ablation_study_results_processed.csv"
df = pd.read_csv(processed_file)

rule_to_operator_mapping = {
    "R_G1": "historically_duration_severity_full_last",
    "R_G2": "historically_duration_severity_full_last",
    "R_G3": "historically_full_last",
    "R_G4": "historically_duration_severity_full_last",
    "R_I1": "historically_duration_full_last",
    "R_I2": "historically_duration_severity_full_last",
    "R_I3": "historically_duration_full_last",
    "R_I4": "historically_duration_full_last",
    "R_I5": "historically_duration_full_last",
}

# ablation study:
# a) model-free, standard stl
# b) model-predictive, standard stl
# c) model-free, enhanced stl
# d) model-predictive, enhanced stl

results = defaultdict(list)  # mfr - mpr
for i, row_mfr in df[~df["mpr"]].iterrows():  # iterate over mfr rows
    row_mpr = df[
        (df["scenario_id"] == row_mfr["scenario_id"])
        & (df["rule"] == row_mfr["rule"])
        & df["mpr"]
        & (df["vehicle_id"] == row_mfr["vehicle_id"])
    ]
    if len(row_mpr) < 1:
        _LOGGER.warning(f"no matching mpr found {i}. Skipping.")
        continue
    assert len(row_mpr) == 1
    row_mpr = row_mpr.iloc[0]
    if row_mfr["historically_full_last"] < 0:
        if np.isnan(row_mfr["historically_full_last"]) or np.isnan(row_mpr["historically_full_last"]):
            _LOGGER.warning("Skipping nan value.")
            continue
        # assert row_mpr["historically_full_last"] < 0, f"{row_mpr["historically_full_last"]}"
        results[(row_mfr["rule"], "a")] += [row_mfr["historically_full_last"]]
        results[(row_mpr["rule"], "b")] += [row_mpr["historically_full_last"]]
        results[(row_mfr["rule"], "c")] += [row_mfr[rule_to_operator_mapping[row_mfr["rule"]]]]
        results[(row_mpr["rule"], "d")] += [row_mpr[rule_to_operator_mapping[row_mpr["rule"]]]]

rows = []
for rule in ["R_G1", "R_G2", "R_G3", "R_G4", "R_I1", "R_I2", "R_I3", "R_I4", "R_I5"]:
    a = results[(rule, "a")]
    b = results[(rule, "b")]
    c = results[(rule, "c")]
    d = results[(rule, "d")]

    assert len(a) == len(b) == len(c) == len(d)

    try:
        rows.append(
            {
                "rule": rule,
                "a span": np.ptp(a),
                "a std": np.std(a),
                "b span": np.ptp(b),
                "b std": np.std(b),
                "c span": np.ptp(c),
                "c std": np.std(c),
                "d span": np.ptp(d),
                "d std": np.std(d),
            }
        )
    except ValueError:
        _LOGGER.warning(f"No traffic rule violation for {rule}")

df_results = pd.DataFrame(rows)

print(df_results)
