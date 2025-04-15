from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

processed_file = Path(__file__).parents[1] / "output" / "ablation_study_results_processed_mfr.csv"
df = pd.read_csv(processed_file)

rule_to_operator_mapping = {
    "R_G1": "historically_duration_severity_",
    "R_G2": "historically_duration_severity_",
    "R_G3": "historically_",
    "R_G4": "historically_duration_severity_",
    "R_I1": "historically_duration_",
    "R_I2": "historically_duration_severity_",
    "R_I3": "historically_duration_",
    "R_I4": "historically_duration_",
    "R_I5": "historically_duration_",
}

duration = "2"

all_traces = True

# ablation study:
# a) model-free, standard stl
# b) model-predictive, standard stl
# c) model-free, enhanced stl
# d) model-predictive, enhanced stl

results = defaultdict(list)  # mfr - mpr
skipped_rectification = 0
skipped_rectification_by_rule = defaultdict(int)
skipped_no_mfr = 0
skipped_positive_only = 0
skipped_nan = 0
for i, row_mfr in df[~df["mpr"]].iterrows():
    # rob_mpr = list(map(float, row_mpr["robustness"].split(",")))
    # if any([x in [-1e-3, 1e-3] for x in rob_mpr]):
    #     # robustness score likely affected by rectification
    #     skipped_rectification += 1
    #     skipped_rectification_by_rule[row_mpr["rule"]] += 1
    #     continue
    # row_mfr = df[
    #     (df["scenario_id"] == row_mpr["scenario_id"])
    #     & (df["rule"] == row_mpr["rule"])
    #     & (~df["mpr"])
    #     & (df["vehicle_id"] == row_mpr["vehicle_id"])
    # ]
    # if len(row_mfr) < 1:
    #     # no corresponding mfr found
    #     skipped_no_mfr += 1
    #     continue
    # assert len(row_mfr) == 1
    # row_mfr = row_mfr.iloc[0]
    if np.isnan(row_mfr[f"historically_{duration}_last"]):
        skipped_nan += 1
        continue
    if all_traces or row_mfr["historically_full_last"] < 0:
        # assert row_mpr["historically_full_last"] < 0, f"{row_mpr["historically_full_last"]}"
        results[(row_mfr["rule"], "a")] += [row_mfr[f"historically_{duration}_min"]] # , row_mfr[f"historically_{duration}_max"]]
        # results[(row_mpr["rule"], "b")] += [row_mpr[f"historically_{duration}_min"], row_mpr[f"historically_{duration}_max"]]
        results[(row_mfr["rule"], "c")] += [row_mfr[rule_to_operator_mapping[row_mfr["rule"]] + f"{duration}_min"]] # , row_mfr[rule_to_operator_mapping[row_mfr["rule"]] + f"{duration}_max"]]
        # results[(row_mpr["rule"], "d")] += [row_mpr[rule_to_operator_mapping[row_mpr["rule"]] + f"{duration}_min"], row_mpr[rule_to_operator_mapping[row_mpr["rule"]] + f"{duration}_max"]]
    else:
        skipped_positive_only += 1

rows = []
for rule in ["R_G1", "R_G2", "R_G3", "R_G4", "R_I1", "R_I2", "R_I3", "R_I4", "R_I5"]:
    a = results[(rule, "a")]
    b = results[(rule, "b")]
    c = results[(rule, "c")]
    d = results[(rule, "d")]

    # assert len(a) == len(b) == len(c) == len(d)

    try:
        rows.append(
            {
                "rule": rule,
                "a span": np.ptp(a),
                "a std": np.std(a),
                #"b span": np.ptp(b),
                #"b std": np.std(b),
                "c span": np.ptp(c),
                "c std": np.std(c),
                #"d span": np.ptp(d),
                #"d std": np.std(d),
                "number": len(a)
            }
        )
    except ValueError:
        _LOGGER.warning(f"No traffic rule violation for {rule}")

df_results = pd.DataFrame(rows)

print(df_results)
