import logging
import multiprocessing
from pathlib import Path

from crmonitor.mpr.learning import DataGenerator
from crmonitor.common import ScenarioType
from crmonitor.predicates import (
    ALL_GENERAL_PREDICATE_NAMES,
    ALL_INTERSTATE_PREDICATE_NAMES,
    CHANGED_TO_META_PREDICATE_NAMES,
)
from crmonitor.predicates.predicate_factory import PredicateFactory

# Use 'all_general_predicates' to generate learning data for all predicates that are used for general traffic rules.
# Alternatively, supply a list of specific predicates you want to evaluate.
predicate_names = (
    ALL_GENERAL_PREDICATE_NAMES + ALL_INTERSTATE_PREDICATE_NAMES + CHANGED_TO_META_PREDICATE_NAMES
)
scenarios_load_path = Path(__file__).parents[3] / "scenarios-for-semantic-aware-stl" / "highD"

output_path = Path(__file__).parents[2] / "output" / "learning_data" / "learning_data.csv"
# Optional: Limit the number of scenarios that are processed e.g. for faster prototyping
scenario_limit = 1

logging.basicConfig(level=logging.DEBUG)
_LOGGER = logging.getLogger(__name__)

predicate_factory = PredicateFactory()
predicate_evaluators = [
    predicate_factory.get_predicate(predicate_name) for predicate_name in predicate_names
]

data_generator = DataGenerator(
    predicates=predicate_evaluators,
    scenarios_path=scenarios_load_path,
    dt=0.04,
    output_path=output_path,
    vehicle_pair_steps=100,
    state_sampling_time_horizon=1.5,
    time_steps_per_scenario=1,
    scenario_type=ScenarioType.INTERSTATE,
    snapshot_frequency=1,
)


_LOGGER.info(
    "Started processing scenarios from %s to generate learning data for predicates %s",
    scenarios_load_path,
    ", ".join([predicate_name for predicate_name in predicate_names]),
)

_LOGGER.info(f"Number of CPUs: {multiprocessing.cpu_count()}")

data_generator.generate_data(workers=1, limit=scenario_limit)  # multiprocessing.cpu_count()
_LOGGER.info("Finished processing scenarios; writing output to %s", output_path)
data_generator.save_data(output_path)
