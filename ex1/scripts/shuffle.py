from typing import Tuple
import numpy as np


def shuffle_dataset(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    indices_shuffled = np.arange(len(y))
    np.random.shuffle(indices_shuffled)
    while len(set(y[:5])) == 1:
        np.random.shuffle(indices_shuffled)
        x, y = x[indices_shuffled], y[indices_shuffled]

    return x, y
