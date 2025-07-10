import logging
from pathlib import Path

from crmonitor.mpr import DataLoader, ModelTrainer
from crmonitor.common import ScenarioType
from crmonitor.predicates.acceleration import PredAbruptBreaking, PredAbruptBreakingRelative
from crmonitor.predicates.base import AbstractPredicate
from crmonitor.predicates.position import (
    PredInFrontOf,
    PredLatCloseToVehicleLeft,
    PredLatCloseToVehicleRight,
)
from crmonitor.mpr.learning.feature_variables import (
    AccelerationFeatureVariable,
    DesiredFeatureVariables,
    FeatureVariableAgentCombination,
    JerkFeatureVariable,
    LateralPositionFeatureVariable,
    OrientationFeatureVariable,
    RelativeDistanceFeatureVariable,
    RelativeLateralDistanceFeatureVariable,
    RelativeLateralVelocityFeatureVariable,
    RelativeLongitudinalVelocityFeatureVariable,
    VehicleLengthFeatureVariable,
    VehicleWidthFeatureVariable,
    VelocityFeatureVariable,
)
from crmonitor.predicates.velocity import (
    PredDrivesWithSlightlyHigherSpeed,
    PredHasQueueVelocity,
    PredHasSlowMovingVelocity,
    PredVelocityBelow15,
    PredVelocityBelow2,
    PredVelocityBelow20,
    PredVelocityBelow5,
)

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

output_folder_path = Path(__file__).parents[2] / "output"

learning_data_path = output_folder_path / "learning_data" / "learning_data.csv"
models_output_path = output_folder_path / "models"

predicates = [PredInFrontOf]

FEATURE_OVERRIDES: dict[type[AbstractPredicate], DesiredFeatureVariables] = {
    PredDrivesWithSlightlyHigherSpeed: {
        FeatureVariableAgentCombination.EGO: [
            VelocityFeatureVariable,
            AccelerationFeatureVariable,
        ],
        FeatureVariableAgentCombination.OTHER: {
            VelocityFeatureVariable,
            AccelerationFeatureVariable,
        },
        FeatureVariableAgentCombination.EGO_OTHER: {
            RelativeLongitudinalVelocityFeatureVariable,
            RelativeLateralDistanceFeatureVariable,
            RelativeDistanceFeatureVariable,
        },
    },
    PredVelocityBelow2: {FeatureVariableAgentCombination.EGO: [VelocityFeatureVariable]},
    PredVelocityBelow5: {FeatureVariableAgentCombination.EGO: [VelocityFeatureVariable]},
    PredVelocityBelow15: {FeatureVariableAgentCombination.EGO: [VelocityFeatureVariable]},
    PredVelocityBelow20: {FeatureVariableAgentCombination.EGO: [VelocityFeatureVariable]},
    PredHasSlowMovingVelocity: {FeatureVariableAgentCombination.EGO: [VelocityFeatureVariable]},
    PredHasQueueVelocity: {
        FeatureVariableAgentCombination.EGO: [VelocityFeatureVariable, AccelerationFeatureVariable]
    },
    PredAbruptBreakingRelative: {
        FeatureVariableAgentCombination.EGO: [
            VelocityFeatureVariable,
            AccelerationFeatureVariable,
            JerkFeatureVariable,
        ],
        FeatureVariableAgentCombination.OTHER: {
            VelocityFeatureVariable,
            AccelerationFeatureVariable,
            JerkFeatureVariable,
        },
        FeatureVariableAgentCombination.EGO_OTHER: {
            RelativeLongitudinalVelocityFeatureVariable,
        },
    },
    PredAbruptBreaking: {
        FeatureVariableAgentCombination.EGO: [
            VelocityFeatureVariable,
            AccelerationFeatureVariable,
            JerkFeatureVariable,
        ],
    },
    PredLatCloseToVehicleLeft: {
        FeatureVariableAgentCombination.EGO: [
            LateralPositionFeatureVariable,
            VehicleWidthFeatureVariable,
            VehicleLengthFeatureVariable,
            OrientationFeatureVariable,
            VelocityFeatureVariable,
        ],
        FeatureVariableAgentCombination.OTHER: {
            LateralPositionFeatureVariable,
            VehicleWidthFeatureVariable,
            VehicleLengthFeatureVariable,
            OrientationFeatureVariable,
            VelocityFeatureVariable,
        },
        FeatureVariableAgentCombination.EGO_OTHER: {
            RelativeLateralDistanceFeatureVariable,
            RelativeLateralVelocityFeatureVariable,
        },
    },
    PredLatCloseToVehicleRight: {
        FeatureVariableAgentCombination.EGO: [
            LateralPositionFeatureVariable,
            VehicleWidthFeatureVariable,
            VehicleLengthFeatureVariable,
            OrientationFeatureVariable,
            VelocityFeatureVariable,
        ],
        FeatureVariableAgentCombination.OTHER: {
            LateralPositionFeatureVariable,
            VehicleWidthFeatureVariable,
            VehicleLengthFeatureVariable,
            OrientationFeatureVariable,
            VelocityFeatureVariable,
        },
        FeatureVariableAgentCombination.EGO_OTHER: {
            RelativeLateralDistanceFeatureVariable,
            RelativeLateralVelocityFeatureVariable,
        },
    },
}


data_loader = DataLoader.create_from_file(learning_data_path)

trainer = ModelTrainer(data_loader, ScenarioType.INTERSTATE, training_iter=400)

for predicate in predicates:
    custom_features = FEATURE_OVERRIDES.get(predicate)

    if custom_features is not None:
        _LOGGER.info(
            f"Training model for predicate {predicate.predicate_name} with custom features."
        )
    else:
        _LOGGER.info(
            f"Training model for predicate {predicate.predicate_name} with default features."
        )
    model_container = trainer.train_predicate(predicate, custom_features)

    model_container.write_to_folder(models_output_path)
