import logging
import warnings
from collections.abc import Iterable

import gpytorch
import numpy as np
import torch

from crmonitor.common import ScenarioType
from crmonitor.mpr.learning.feature_extractor import FeatureExtractor
from crmonitor.predicates import AbstractPredicate

from ._split_data import split_data
from .data_loader import DataLoader
from .exact_gp_model import ExactGPModel, ExactGPModelContainer, ExactGPModelContainerVersion
from .feature_variables import (
    DesiredFeatureVariables,
)

_LOGGER = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(
        self,
        data_loader: DataLoader,
        scenario_type: ScenarioType = ScenarioType.INTERSTATE,
        training_iter: int = 100,
    ) -> None:
        self._scenario_type: ScenarioType = scenario_type
        self._data_loader: DataLoader = data_loader
        self._training_iter = training_iter

    def train(
        self, predicates: Iterable[type[AbstractPredicate]]
    ) -> dict[str, ExactGPModelContainer]:
        models: dict[str, ExactGPModelContainer] = {}
        for predicate in predicates:
            res = self.train_predicate(predicate)
            models[predicate.predicate_name] = res
        return models

    def train_predicate(
        self,
        predicate: type[AbstractPredicate],
        desired_features: DesiredFeatureVariables | None = None,
    ) -> ExactGPModelContainer:
        _LOGGER.info(
            "training predicate %s for scenario type %s",
            predicate.predicate_name,
            self._scenario_type,
        )
        if desired_features is None:
            feature_extractor = FeatureExtractor.for_scenario_type(self._scenario_type)
        else:
            feature_extractor = FeatureExtractor(desired_features)
        index_features = feature_extractor.feature_variable_labels(predicate)

        X, y = self._data_loader.Xy(predicate, index_features)
        # 4:1 split
        X_train, _, y_train, _ = split_data(X, y)
        # train GPR
        train_X = torch.Tensor(X_train.astype(np.float64))  # TODO would np.float32 be enough?
        train_y = torch.Tensor(y_train.astype(np.float64))
        model = self._train_model(train_X, train_y)

        return ExactGPModelContainer(
            version=ExactGPModelContainerVersion.VERSION_1_0,
            name=predicate.predicate_name,
            model=model,
            features=feature_extractor.desired_feature_variables,
            scenario_type=self._scenario_type,
        )

    def _train_model(self, train_X: torch.Tensor, train_y: torch.Tensor) -> ExactGPModel:
        model = ExactGPModel.create_new_model(train_X, train_y, self._scenario_type)

        _ = model.train()

        if self._scenario_type == ScenarioType.INTERSECTION:
            train_X = train_X.double()
            train_y = train_y.double()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.1
        )  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

        # warnings.filterwarnings("ignore")
        for i in range(self._training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_X)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            (i + 1) % 10 == 0 and _LOGGER.info(
                f"Iter {i + 1}/{self._training_iter} - Loss: {loss.item():.3f} - Noise: {sum(model.likelihood.noise).item():.3f}"
            )
            optimizer.step()
            # if loss.item() < -1:
            #     _LOGGER.info(f"Preempting training after {i+1} iterations, as loss is {loss.item():.3f}.")
            #     break

        warnings.filterwarnings("default")
        return model
