from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
from matplotlib.figure import Figure
from sklearn.metrics import mean_squared_error

from crmonitor.predicates import AbstractPredicate

from ._split_data import split_data
from .data_loader import DataLoader
from .exact_gp_model import ExactGPModel, ExactGPModelContainer, read_model
from .feature_extractor import FeatureExtractor


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics of a single predicate."""

    y_test: npt.NDArray[np.float64]
    y_pred: npt.NDArray[np.float64]
    size: int
    bool_pred: npt.NDArray[np.bool_]
    bool_gt: npt.NDArray[np.bool_]
    TP: int
    FP: int
    FN: int
    TN: int
    precision: float
    recall: float
    f1_score: float
    mse: float
    ratio_large_std: float
    span_train: float
    span_test: float
    span_pred: float
    span_mfr: float
    std_train: float
    std_test: float
    std_pred: float
    std_mfr: float
    mean_train: float
    mean_test: float
    mean_pred: float
    mean_mfr: float
    shap_values: None = None


class ModelEvaluator:
    def __init__(
        self,
        data_loader: DataLoader,
        models_path: Path,
        eps: float = 1e-9,
    ) -> None:
        self._data_loader = data_loader
        self._models_path = models_path
        self._eps = eps

    def evaluate(
        self, predicates: Iterable[type[AbstractPredicate]]
    ) -> dict[str, EvaluationMetrics]:
        """plot the comparison between ground truth and prediction,
        and calculate mean squared error for all predicate models.

        Returns:
            Tuple[Figure, Axes, List[float]]: [fig, ax, mse]
        """

        results: dict[str, EvaluationMetrics] = dict()

        for predicate in predicates:
            results[predicate.predicate_name] = self.evaluate_predicate(predicate)

        return results

    def evaluate_predicate(self, predicate: type[AbstractPredicate]) -> EvaluationMetrics:
        """Evaluate a single predicate and return its metrics.

        Args:
            predicate: The predicate evaluator to evaluate.

        Returns:
            EvaluationMetrics object containing all computed metrics.
        """
        # Load model and data
        model_container = read_model(predicate.predicate_name, self._models_path)

        feature_extractor = FeatureExtractor(model_container.features)
        index_features = feature_extractor.feature_variable_labels(predicate)
        X, y, y_mfr = self._data_loader.Xy(predicate, index_features, mfr_data=True)

        # Split data
        X_train, X_test, y_train, y_test, y_mfr_train, y_mfr_test = split_data(X, y, y_mfr)

        # Make predictions
        y_pred, std = self._make_predictions(model_container.model, X_test)
        num_large_std = sum(1 for s in std if s > 0.1)
        ratio_large_std = num_large_std / len(std)

        # Calculate metrics
        mse: float = mean_squared_error(y_test, y_pred)
        bool_pred = (y_pred > 0).flatten()
        bool_gt = (y_test > 0).flatten()

        true_positive = np.logical_and(bool_pred, bool_gt).sum()
        false_positive = np.logical_and(bool_pred, ~bool_gt).sum()
        false_negative = np.logical_and(~bool_pred, bool_gt).sum()
        true_negative = np.logical_and(~bool_pred, ~bool_gt).sum()

        precision = (true_positive + self._eps) / (true_positive + false_positive + self._eps)
        recall = (true_positive + self._eps) / (true_positive + false_negative + self._eps)
        f1_score = 2 * precision * recall / (precision + recall)

        # Create metrics object
        metrics = EvaluationMetrics(
            y_test=np.asarray(y_test, dtype=float),
            y_pred=np.asarray(y_pred, dtype=float),
            size=len(y_test),
            bool_pred=bool_pred,
            bool_gt=bool_gt,
            TP=true_positive,
            FP=false_positive,
            FN=false_negative,
            TN=true_negative,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            mse=mse,
            ratio_large_std=ratio_large_std,
            span_train=np.ptp(y_train, axis=0),
            span_test=np.ptp(y_test, axis=0),
            span_pred=np.ptp(y_pred, axis=0)[0],
            span_mfr=np.ptp(y_mfr_test, axis=0),
            std_train=np.std(y_train, axis=0),
            std_test=np.std(y_test, axis=0),
            std_pred=np.std(y_pred, axis=0)[0],
            std_mfr=np.std(y_mfr_test, axis=0),
            mean_train=np.mean(y_train, axis=0),
            mean_test=np.mean(y_test, axis=0),
            mean_pred=np.mean(y_pred, axis=0)[0],
            mean_mfr=np.mean(y_mfr_test, axis=0),
        )

        # Add SHAP analysis if enabled
        run_shap = False  # This could be a parameter
        if run_shap:
            shap_values = self._calculate_shap_values(model_container, None, predicate)
            metrics.shap_values = shap_values

        return metrics

    def _make_predictions(
        self, model: ExactGPModel, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Make predictions using the model and clip results."""
        y_pred, std = model.predict(X_test.astype(float))
        y_pred = np.clip(y_pred, -1, 1)
        return y_pred, std

    def _create_balanced_test_set(
        self, X_test: np.ndarray, y_test: np.ndarray, n_samples_each: int
    ) -> np.ndarray:
        """Create a balanced test set for SHAP analysis."""
        indices_true = np.where(y_test > 0)[0]
        indices_false = np.where(y_test <= 0)[0]

        if len(indices_true) > n_samples_each:
            indices_true = np.random.choice(indices_true, n_samples_each, replace=False)
        if len(indices_false) > n_samples_each:
            indices_false = np.random.choice(indices_false, n_samples_each, replace=False)

        balanced_indices = np.concatenate([indices_true, indices_false])
        return X_test[balanced_indices].astype(np.float32)

    def _calculate_shap_values(
        self,
        model_container: ExactGPModelContainer,
        data_split: Any,
        predicate: type[AbstractPredicate],
        n_samples_each: int = 100,
    ) -> Any:
        """Calculate SHAP values for feature importance analysis."""
        # Balance the test set for SHAP analysis
        X_test_balanced = self._create_balanced_test_set(
            data_split.X_test, data_split.y_test, n_samples_each
        )

        feature_extractor = FeatureExtractor(model_container.features)
        index_features = feature_extractor.feature_variable_labels(predicate)

        # Calculate SHAP values
        shap_explainer = shap.Explainer(
            lambda x: model_container.model.predict(x)[0], data_split.X_train.astype(np.float32)
        )
        shap_values = shap_explainer(X_test_balanced)

        # Set feature names
        feature_names = [", ".join(feature) for feature in index_features]
        shap_values.feature_names = feature_names

        return shap_values

    def visualize_shap(self, results: dict[str, EvaluationMetrics]) -> tuple[Figure, np.ndarray]:
        """Visualize SHAP values for all predicates with available SHAP data.

        Args:
            results: Dictionary of evaluation results.

        Returns:
            Tuple of matplotlib Figure and axes array.
        """
        # Filter results that have SHAP values
        shap_results = {
            name: metrics for name, metrics in results.items() if metrics.shap_values is not None
        }

        if not shap_results:
            raise ValueError("No SHAP values found in results")

        rows = len(shap_results)
        fig, axs = plt.subplots(rows, 1, figsize=(10, rows * 4), squeeze=False, tight_layout=True)

        for i, (predicate_name, metrics) in enumerate(shap_results.items()):
            shap_ax = axs[i, 0]
            shap_ax.set_title(f"{predicate_name}")
            shap.plots.bar(metrics.shap_values, ax=shap_ax, show=False)

        plt.autoscale()
        return fig, axs

    def visualize_prediction(
        self, results: dict[str, EvaluationMetrics]
    ) -> tuple[Figure, np.ndarray]:
        rows = len(results)
        fig, axs = plt.subplots(rows, 1, figsize=(10, rows * 4), squeeze=False, tight_layout=True)

        for i, (predicate_name, data) in enumerate(results.items()):
            pred_ax = axs[i, 0]
            y_test = data.y_test
            y_pred = data.y_pred

            y_test = np.asarray(y_test, dtype=float).flatten()
            y_pred = np.asarray(y_pred, dtype=float).flatten()

            pred_ax.scatter(y_test, y_pred, s=1)
            pred_ax.plot([-1, 1], [-1, 1], "--k", alpha=0.2)
            pred_ax.set_xlabel("Ground Truth")
            pred_ax.set_ylabel("Prediction")
            pred_ax.set_title(
                f"{predicate_name}\nF1: {data.f1_score:.3f}, MSE: {data.mse:.3e}, #test samples: {len(y_test)}"
            )

        plt.autoscale()
        # tikzplotlib.save(f"/tmp/plots/figure.tikz")
        return fig, axs

    def save(self, results: dict[str, EvaluationMetrics], file_path: Path) -> None:
        """Save evaluation results to CSV file.

        Args:
            results: Dictionary of evaluation results.
            file_path: Path where to save the CSV file.
        """
        columns_to_exclude = {"shap_values", "bool_gt", "bool_pred", "y_test", "y_pred"}

        flattened_results = []
        for predicate_name, metrics in results.items():
            # Convert dataclass to dictionary
            metrics_dict = metrics.__dict__.copy()

            # Add predicate name and filter out excluded columns
            filtered_dict = {"predicate": predicate_name}
            for key, value in metrics_dict.items():
                if key not in columns_to_exclude:
                    filtered_dict[key] = value

            flattened_results.append(filtered_dict)

        results_df = pd.DataFrame(flattened_results)
        results_df.to_csv(file_path, index=False)
