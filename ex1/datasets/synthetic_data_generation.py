import numpy as np
from typing import Tuple


def create_dataset_by_dimention(dim: int) -> Tuple[np.ndarray, np.ndarray]:
    mean1 = np.ones(dim) / np.sqrt(dim)
    mean2 = - mean1
    cov_mat1 = cov_mat2 = np.eye(dim)
    return create_dataset_of_two_gaussians(mean1=mean1, cov_mat1=cov_mat1, mean2=mean2, cov_mat2=cov_mat2,
                                           n_samples=600)


def create_dataset_of_two_gaussians(mean1: np.ndarray, cov_mat1: np.ndarray,
                                    mean2: np.ndarray, cov_mat2: np.ndarray,
                                    n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    n_samples_per_label = n_samples // 2

    x_pos = sample_from_normal_distribution(mean1, cov_mat1, n_samples=n_samples_per_label)
    x_neg = sample_from_normal_distribution(mean2, cov_mat2, n_samples=n_samples_per_label)

    y_pos = np.ones(n_samples_per_label)
    y_neg = - np.ones(n_samples_per_label)
    x = np.vstack([x_pos, x_neg])
    y = np.hstack([y_pos, y_neg])

    return x, y


def sample_from_normal_distribution(mean: np.ndarray, cov_mat: np.ndarray, n_samples: int) -> np.ndarray:
    return np.random.multivariate_normal(mean=mean, cov=cov_mat, size=n_samples)


def create_poor_performance_dataset_for_simple():
    """simple is expected to perform poorly on unbalanced datasets"""
    x = np.random.randn(600, 1)
    y = (x > 2).flatten().astype(int)
    return x, y


