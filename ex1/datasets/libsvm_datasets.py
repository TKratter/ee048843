from typing import Tuple, Callable
from sklearn.datasets import load_breast_cancer, load_svmlight_file
from sklearn.preprocessing import normalize
import numpy as np


def get_breast_cancer_data_and_labels() -> Tuple[np.ndarray, np.ndarray]:
    bunch = load_breast_cancer()
    x = normalize(bunch.data)
    y = bunch.target
    return x, y


def get_diabetes_data_and_labels() -> Tuple[np.ndarray, np.ndarray]:
    return load_svmlight_file('/home/tomk42/PycharmProjects/ee048843/ex1/datasets/diabetes_scaled.txt')


def _get_data_and_labels_from_loader_func(loader_func: Callable) -> Tuple[np.ndarray, np.ndarray]:
    bunch = loader_func()
    x = bunch.data
    y = bunch.target
    return x, y
