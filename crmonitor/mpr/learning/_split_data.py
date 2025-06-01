import logging
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import train_test_split

_LOGGER = logging.getLogger(__name__)


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    y_mfr: np.ndarray | None = None,
    train_split: float = 0.8,
    bins: int = 3,
    random_state: Any = 1,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """
    The outputs are divided into [bins] bins.
    If one of the bins has less than [num_train/bins/hidden_test_size] samples,
    we select [hidden_test_size] of the samples in the bin.

    Args:
        X (np.ndarray): Inputs
        y (np.ndarray): Outputs
        y_mfr (np.ndarray): MFR
        train_split (float): Ratio of how many samples are in the training set.
        bins (int, optional): Number of bins (default is 1).
        random_state (Any): Random_state for sklearn.model_selection.train_test_split

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
    """
    assert 0 < train_split < 1, f"train_size should be in (0, 1), got {train_split}."

    use_mfr = False
    if y_mfr is not None:
        assert len(y_mfr) == len(y)
        use_mfr = True

    Xs, ys, ys_mfr = gen_bins(X, y, y_mfr, bins)
    median = int(np.median([len(y_temp) for y_temp in ys]))
    if median < 1000 / bins:
        _LOGGER.warning(
            "Too few samples. Increasing median manually, which causes un-balanced data set."
        )
        median = int(1000 / bins)

    X_train, X_test, y_train, y_test, y_mfr_train, y_mfr_test = [], [], [], [], [], []
    for X_bin, y_bin, y_mfr_bin in zip(Xs, ys, ys_mfr):
        if len(y_bin) == 0:
            continue
        elif len(y_bin) == 1:
            X_train.append(X_bin)
            y_train.append(y_bin)
            y_mfr_train.append(y_mfr_bin)
        else:
            train_size = int(train_split * min(len(y_bin), median))
            test_size = min(len(y_bin), median) - train_size
            X_train_bin, X_test_bin, y_train_bin, y_test_bin, y_mfr_train_bin, y_mfr_test_bin = (
                train_test_split(
                    *(X_bin, y_bin, y_mfr_bin),
                    train_size=train_size,
                    test_size=test_size or None,
                    random_state=random_state,
                )
            )
            X_train.append(X_train_bin)
            X_test.append(X_test_bin)
            y_train.append(y_train_bin)
            y_test.append(y_test_bin)
            y_mfr_train.append(y_mfr_train_bin)
            y_mfr_test.append(y_mfr_test_bin)

    X_train = np.concatenate(X_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    y_mfr_train = np.concatenate(y_mfr_train, axis=0)
    y_mfr_test = np.concatenate(y_mfr_test, axis=0)

    _LOGGER.info(
        f"Got {len(y_train)} training samples and {len(y_test)} testing samples from {len(y)} samples."
    )
    for name, y_vec in zip(("Train", "Test"), (y_train, y_test)):
        y_true_len = len([y for y in y_vec if y > 0])
        y_false_len = len([y for y in y_vec if y < 0])
        _LOGGER.info(
            f"{name} set: mean={np.mean(y_vec):.3f}, std={np.std(y_vec):.3f}, span={np.ptp(y_vec):.3f}, balance={(y_true_len - y_false_len) / (y_true_len + y_false_len + 1e-7):.3f}"
        )

    if use_mfr:
        return X_train, X_test, y_train, y_test, y_mfr_train, y_mfr_test
    else:
        return X_train, X_test, y_train, y_test


def gen_bins(
    X: np.ndarray,
    y: np.ndarray,
    y_mfr: Optional[np.ndarray] = None,
    bins: int = 1,
) -> Tuple[List[np.ndarray], List[np.ndarray], Union[List[np.ndarray], None]]:
    Xs = []
    ys = []
    ys_mfr = []
    use_mfr = False
    if y_mfr is not None:
        assert len(y_mfr) == len(y)
        use_mfr = True
    bin_edges = np.linspace(y.min(), y.max(), bins + 1)[1:]
    bin_ids = np.searchsorted(bin_edges, y)
    for bin_id in range(bins):
        Xs.append(X[bin_ids == bin_id])
        ys.append(y[bin_ids == bin_id])
        if use_mfr:
            ys_mfr.append(y_mfr[bin_ids == bin_id])
        else:
            ys_mfr.append(np.full(len(y[bin_ids == bin_id]), np.nan))
    return Xs, ys, ys_mfr
