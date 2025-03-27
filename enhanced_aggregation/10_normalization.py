import logging
from pathlib import Path

from commonroad_mpr.learning import DataLoader
from commonroad_mpr.learning.data_loader import normalize
from crmonitor.predicate_grouping import all_general_predicates, all_interstate_predicates

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

learning_data_path = Path(__file__).parent.parent / "output" / "learning_data" / "learning_data.csv"

selected_predicates = all_general_predicates + all_interstate_predicates
data_loader = DataLoader.create_from_file(learning_data_path)

# check mean, std, and span
for predicate in selected_predicates:
    data_rob_raw = data_loader.data[("predicates", predicate, "robustness")]
    data_rob_nor = normalize(data_loader.data[("predicates", predicate, "robustness")])

    # data_rob_nor = data_loader.data[("predicates", predicate, "normalized_robustness")]
    for i, rob_raw, rob_nor in zip(range(len(data_rob_raw)), data_rob_raw, data_rob_nor):
        assert abs(rob_raw) <= abs(rob_nor), (
            f"Predicate {predicate}, row {i} has raw robustness {rob_raw:.3f} and normalized robustness {rob_nor:.3f}"
        )
