from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pytest
import torch
from crmonitor.common import ScenarioType
from crmonitor.mpr.learning.exact_gp_model import (
    ExactGPModel,
    ExactGPModelContainer,
    ExactGPModelContainerVersion,
    read_model,
)
from crmonitor.mpr.learning.feature_variables import (
    AccelerationFeatureVariable,
    DesiredFeatureVariables,
    FeatureVariableAgentCombination,
    RelativeDistanceFeatureVariable,
    RelativeLateralDistanceFeatureVariable,
    RelativeLongitudinalVelocityFeatureVariable,
    VehicleLengthFeatureVariable,
    VelocityFeatureVariable,
)


class TestExactGpModelContainer:
    @pytest.fixture
    def mock_exact_gp_model(self):
        """Create a mock ExactGPModel."""
        model = Mock(spec=ExactGPModel)
        model.train_inputs = [torch.randn(10, 2)]
        model.train_targets = torch.randn(10)
        model.state_dict.return_value = {"param1": torch.randn(5, 5)}
        model.load_state_dict = Mock()
        return model

    @pytest.mark.parametrize("scenario_type", [ScenarioType.INTERSTATE, ScenarioType.INTERSECTION])
    @pytest.mark.parametrize(
        "features",
        [
            {FeatureVariableAgentCombination.EGO: [VelocityFeatureVariable]},
            {
                FeatureVariableAgentCombination.EGO: [
                    AccelerationFeatureVariable,
                    VehicleLengthFeatureVariable,
                ],
                FeatureVariableAgentCombination.OTHER: [
                    AccelerationFeatureVariable,
                    VehicleLengthFeatureVariable,
                ],
                FeatureVariableAgentCombination.EGO_OTHER: [
                    RelativeDistanceFeatureVariable,
                    RelativeLongitudinalVelocityFeatureVariable,
                ],
                FeatureVariableAgentCombination.OTHER_EGO: [RelativeLateralDistanceFeatureVariable],
            },
        ],
    )
    def test_written_model_can_be_read_back(
        self, scenario_type: ScenarioType, features: DesiredFeatureVariables, mock_exact_gp_model
    ):
        predicate_name = "test_predicate"

        orig_container = ExactGPModelContainer(
            version=ExactGPModelContainerVersion.VERSION_1_0,
            name=predicate_name,
            model=mock_exact_gp_model,
            features=features,
            scenario_type=scenario_type,
        )

        with TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / f"{predicate_name}.p"
            orig_container.write_to_file(temp_file)

            assert temp_file.exists()

            loaded_container = ExactGPModelContainer.read_from_file(temp_file)

            assert loaded_container.name == orig_container.name
            assert loaded_container.features == orig_container.features
            assert loaded_container.scenario_type == orig_container.scenario_type


class TestReadModel:
    @pytest.mark.parametrize(
        "predicate_name", ["in_front_of", "velocity_below_five", "brakes_abruptly", "on_shoulder"]
    )
    def test_can_load_model_from_repo(self, predicate_name: str):
        model_container = read_model(predicate_name)
        assert model_container.name == predicate_name
