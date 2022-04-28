from typing import Tuple, Callable
from sklearn.datasets import load_breast_cancer, load_diabetes
import numpy as np


def get_breast_cancer_data_and_labels() -> Tuple[np.ndarray, np.ndarray]:
    return _get_data_and_labels_from_loader_func(load_breast_cancer)


def get_diabetes_data_and_labels() -> Tuple[np.ndarray, np.ndarray]:
    return _get_data_and_labels_from_loader_func(load_diabetes)


def _get_data_and_labels_from_loader_func(loader_func: Callable) -> Tuple[np.ndarray, np.ndarray]:
    bunch = loader_func()
    x = bunch.data
    y = bunch.target
    return x, y
