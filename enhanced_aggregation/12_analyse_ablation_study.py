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
for _, row_mfr in df[~df["mpr"]].iterrows():  # iterate over mfr rows
    row_mpr = df[(df["scenario_id"] == row_mfr["scenario_id"]) & (df["rule"] == row_mfr["rule"]) & df["mpr"] & (df["vehicle_id"] == row_mfr["vehicle_id"])]
    assert len(row_mpr) == 1
    row_mpr = row_mpr.iloc[0]
    if row_mfr["historically_full_last"] < 0:
        results[(row_mfr["rule"], "a")] += [row_mfr["historically_full_last"]]
    if row_mpr["historically_full_last"] < 0:
        results[(row_mpr["rule"], "b")] += [row_mpr["historically_full_last"]]
    if row_mfr[rule_to_operator_mapping[row_mfr["rule"]]] < 0:
        results[(row_mfr["rule"], "c")] += [row_mfr[rule_to_operator_mapping[row_mfr["rule"]]]]
    if row_mpr[rule_to_operator_mapping[row_mpr["rule"]]] < 0:
        results[(row_mpr["rule"], "d")] += [row_mpr[rule_to_operator_mapping[row_mpr["rule"]]]]

rows = []
for rule in ["R_G1", "R_G2", "R_G3", "R_G4", "R_I1", "R_I3", "R_I4", "R_I5"]:
    a = results[(rule, "a")]
    b = results[(rule, "b")]
    c = results[(rule, "c")]
    d = results[(rule, "d")]

    # _LOGGER.info(f"{rule}: a span {np.ptp(a):.2f}, std {np.std(a):.2f}")
    # _LOGGER.info(f"{rule}: b span {np.ptp(b):.2f}, std {np.std(b):.2f}")
    # _LOGGER.info(f"{rule}: c span {np.ptp(c):.2f}, std {np.std(c):.2f}")
    # _LOGGER.info(f"{rule}: d span {np.ptp(d):.2f}, std {np.std(d):.2f}")

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

