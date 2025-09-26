import importlib.resources
import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import gpytorch
import numpy as np
import torch

import crmonitor.mpr.models.intersection as pre_trained_intersection_models
import crmonitor.mpr.models.interstate as pre_trained_interstate_models
from crmonitor.common import ScenarioType

from .feature_variables import (
    DesiredFeatureVariables,
    FeatureVariableAgentCombination,
    default_feature_variable_classes_for_scenario_type,
    get_all_available_feature_variables,
)

_DEVICE = torch.device(
    f"cuda:{torch.cuda.device_count() - 2}" if torch.cuda.is_available() else "cpu"
)

_LOGGER = logging.getLogger(__name__)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        scenario_type: ScenarioType,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self._scenario_type = scenario_type
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
        self.last_covar = None

    @classmethod
    def create_new_model(
        cls,
        train_X: torch.Tensor,
        train_y: torch.Tensor,
        scenario_type: ScenarioType = ScenarioType.INTERSTATE,
    ) -> "ExactGPModel":
        fixed_noise_tensor = torch.full(
            train_y.shape,
            0.001,
            dtype=train_y.dtype,
            device=train_y.device,
        )
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=fixed_noise_tensor).to(
            _DEVICE
        )
        if scenario_type == ScenarioType.INTERSECTION:
            # FIXME: for intersection, the model needs to use double, this reduces the speed of the model.
            likelihood = likelihood.double()

        if scenario_type == ScenarioType.INTERSTATE:
            model = cls(train_X, train_y, likelihood, scenario_type)
        else:
            # FIXME: for intersection, the model needs to use double, this reduces the speed of the model.
            model = cls(train_X.double(), train_y.double(), likelihood, scenario_type).double()
        model.to(_DEVICE)

        return model

    def eval(self) -> "ExactGPModel":
        _ = super().eval()
        _ = self.likelihood.eval()
        return self

    def train(self, mode: bool = True) -> "ExactGPModel":
        _ = super().train(mode)
        _ = self.likelihood.train(mode)
        return self

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, X: np.ndarray) -> tuple[float, float]:
        self.eval()
        self.likelihood.eval()
        # TODO: data type
        if self._scenario_type == ScenarioType.INTERSTATE:
            X = torch.Tensor(X).to(_DEVICE)
            # Make sure the model and likelihood are also in float
            self.to(dtype=torch.float)
            self.likelihood = self.likelihood.float()
        else:
            X = torch.Tensor(X).to(_DEVICE).double()
            # Make sure the model and likelihood are also in double
            self.to(dtype=torch.double)
            self.likelihood = self.likelihood.double()

        warnings.filterwarnings("ignore")
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self(X))
            self.last_covar = observed_pred.lazy_covariance_matrix
            y = observed_pred.mean
            std = observed_pred.stddev
        warnings.filterwarnings("default")
        if len(y) == 1:
            return y.detach().item(), std.detach().item()
        else:
            return (
                y.unsqueeze(1).to(_DEVICE).cpu().detach().numpy(),
                std.unsqueeze(1).to(_DEVICE).cpu().detach().numpy(),
            )

    def get_gradient(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure x is a float tensor and requires gradient
        if self._scenario_type == ScenarioType.INTERSTATE:
            X = torch.tensor(x, device=_DEVICE, dtype=torch.float32, requires_grad=True)
        else:
            X = torch.tensor(x, device=_DEVICE, dtype=torch.float64, requires_grad=True)
        # Get the prediction from the model
        observed_pred = self.likelihood(self(X))

        # We sum the mean predictions to get a scalar output
        y = observed_pred.mean.sum()

        # Backward pass to calculate the gradients
        _ = y.backward()
        return X.grad


class ExactGPModelContainerVersion(Enum):
    NO_METADATA = "no_metadata"
    VERSION_1_0 = "1.0"


@dataclass
class ExactGPModelContainer:
    """The ModelContainer is used to save and load GP models with metadata."""

    version: ExactGPModelContainerVersion
    name: str

    model: ExactGPModel

    features: DesiredFeatureVariables
    scenario_type: ScenarioType = ScenarioType.INTERSTATE

    def to_dict(self) -> dict[str, Any]:
        state_dict = self.model.state_dict()
        train_X = self.model.train_inputs[0]
        train_y = self.model.train_targets
        features = self._serialize_desired_feature(self.features)
        return {
            "version": self.version.value,
            "name": self.name,
            "state_dict": state_dict,
            "train_X": train_X,
            "train_y": train_y,
            "features": features,
            "scenario_type": self.scenario_type.value,
        }

    @classmethod
    def from_dict(cls, dict_: dict[str, Any]) -> "ExactGPModelContainer":
        model_name: str = dict_["name"]

        try:
            model_version = ExactGPModelContainerVersion(dict_["version"])
        except ValueError as e:
            raise RuntimeError(
                f"Model version {dict_['version']} of model {model_name} is not supported!"
            ) from e

        try:
            scenario_type = ScenarioType(dict_["scenario_type"])
        except ValueError as e:
            raise RuntimeError(
                f"Scenario type {dict_['scenario_type']} of model {model_name} is not supported!"
            ) from e

        if "features" not in dict_:
            raise ValueError(f"Model of version {model_version} requires field 'features'.")
        features = cls._deserialize_desired_feature(dict_["features"])

        model = ExactGPModel.create_new_model(dict_["train_X"], dict_["train_y"], scenario_type)
        try:
            model.load_state_dict(dict_["state_dict"], strict=False)
        except AttributeError as e:
            _LOGGER.error(f"Failed to load model for predicate '{model_name}': {e}")
            raise Exception

        return cls(
            model_version,
            model_name,
            model,
            features,
            scenario_type,
        )

    def write_to_file(self, file: Path) -> None:
        torch.save(self.to_dict(), file)

    def write_to_folder(self, folder: Path) -> None:
        self.write_to_file(folder / f"{self.name}.p")

    @classmethod
    def read_from_file(cls, file: Path) -> "ExactGPModelContainer":
        model_content = torch.load(file, map_location=_DEVICE, weights_only=False)
        if isinstance(model_content, tuple):
            # Legacy model, without any metadata
            state_dict, train_X, train_y = model_content
            if hasattr(state_dict, "state_dict"):
                state_dict = state_dict.state_dict()

            model = ExactGPModel.create_new_model(train_X, train_y, ScenarioType.INTERSTATE)
            model.load_state_dict(state_dict, strict=False)

            return cls(
                ExactGPModelContainerVersion.NO_METADATA,
                file.stem,
                model,
                default_feature_variable_classes_for_scenario_type(ScenarioType.INTERSTATE),
            )
        else:
            return cls.from_dict(model_content)

    @staticmethod
    def _serialize_desired_feature(
        desired_features: DesiredFeatureVariables,
    ) -> dict[str, list[str]]:
        raw_desired_features = {}

        for (
            agent_combination,
            desired_feature_variables_for_agent_combination,
        ) in desired_features.items():
            raw_desired_feature_variables_for_agent_combination = [
                feature_variable.name
                for feature_variable in desired_feature_variables_for_agent_combination
            ]
            raw_desired_features[agent_combination.value] = (
                raw_desired_feature_variables_for_agent_combination
            )

        return raw_desired_features

    @staticmethod
    def _deserialize_desired_feature(
        raw_desired_features: dict[str, list[str]],
    ) -> DesiredFeatureVariables:
        all_available_features = get_all_available_feature_variables()

        parsed_desired_features = {}
        for (
            raw_agent_combination,
            raw_desired_features_for_agent_combination,
        ) in raw_desired_features.items():
            desired_features_for_agent_combination = []
            for raw_desired_feature in raw_desired_features_for_agent_combination:
                if raw_desired_feature not in all_available_features:
                    raise RuntimeError()

                desired_features_for_agent_combination.append(
                    all_available_features[raw_desired_feature]
                )

            agent_combination = FeatureVariableAgentCombination(raw_agent_combination)
            parsed_desired_features[agent_combination] = desired_features_for_agent_combination

        return parsed_desired_features


class ModelLoadError(Exception):
    """
    Indicates that a pre-trained model for GP evaluation could not be loaded from a specfic model path.
    This is used in downstream libraries to inform users abount missing models.
    """

    def __init__(self, predicate_name: str, model_path: Path) -> None:
        super().__init__(
            f"Failed to load pre-trained model for predicate '{predicate_name}' from path '{model_path}'"
        )
        self.predicate_name = predicate_name
        self.model_path = model_path


_MODELS_CACHE: dict[str, ExactGPModelContainer] = {}


def read_model(
    predicate_name: str,
    model_path: Path | None = None,
    scenario_type: ScenarioType = ScenarioType.INTERSTATE,
) -> ExactGPModelContainer:
    """
    Load a pre-trained model for predicate `predicate_name`.

    :param predicate_name: The name of the predicate. The model file must be named exactly the same.
    :param model_path: Optionally provide a path where models can be found. If None is given, the models from the repo are used.
    :param scenario_type: Specify the scenario type on which the models were trained.

    :returns: The model container.

    :raises ModelLoadError: If the model could not be loaded.
    """
    if predicate_name not in _MODELS_CACHE:
        model_container = _load_model_container(predicate_name, model_path, scenario_type)
        _MODELS_CACHE[predicate_name] = model_container
    return _MODELS_CACHE[predicate_name]


def _load_model_container(
    predicate_name: str,
    model_path: Path | None = None,
    scenario_type: ScenarioType = ScenarioType.INTERSTATE,
) -> ExactGPModelContainer:
    predicate_file_name = f"{predicate_name}.p"
    if model_path is not None:
        predicate_path = model_path / predicate_file_name
        if predicate_path.exists():
            _LOGGER.debug(
                "Loaded pre-trained model for predicate '%s' from '%s'",
                predicate_name,
                predicate_path,
            )
            return ExactGPModelContainer.read_from_file(predicate_path)

    if scenario_type == ScenarioType.INTERSTATE:
        pre_trained_models_path = pre_trained_interstate_models
    else:
        pre_trained_models_path = pre_trained_intersection_models

    with importlib.resources.path(pre_trained_models_path, predicate_file_name) as predicate_path:
        if predicate_path.exists():
            _LOGGER.debug(
                "Loaded pre-trained model for predicate '%s' from '%s'",
                predicate_name,
                predicate_path,
            )
            return ExactGPModelContainer.read_from_file(predicate_path)

        if model_path is None:
            raise ModelLoadError(predicate_name, predicate_path)
        else:
            raise ModelLoadError(predicate_name, model_path)
