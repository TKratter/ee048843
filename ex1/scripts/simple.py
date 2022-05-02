import numpy as np
from typing import List, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import defaultdict
import matplotlib.pyplot as plt

from ex1.config import query_num_list
from ex1.scripts.plot_utils import plot_scores_dict


def perform_simple_algorithm(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                             save_path: Optional[str] = None, title: Optional[str] = None,
                             verbose: Optional[bool] = False) -> Dict[str, List[float]]:
    query_num = 20
    scores_dict = defaultdict(list)
    for kernel in ['linear', 'rbf']:
        current_x_train, current_y_train = x_train[:5], y_train[:5]
        y_train_indices = list(range(5))
        while len(y_train_indices) <= 290:
            clf = SVC(kernel=kernel)
            clf.fit(current_x_train, current_y_train)

            candidates_distances = np.abs(clf.decision_function(x_train))
            indices_sorted = candidates_distances.argsort()
            relevant_indices = [i for i in indices_sorted if i not in y_train_indices]
            new_indices = relevant_indices[:query_num]
            y_train_indices += new_indices
            current_x_train, current_y_train = x_train[y_train_indices], y_train[y_train_indices]

            y_pred = clf.predict(x_test)

            scores_dict[kernel].append(1 - accuracy_score(y_test, y_pred))
    if verbose:
        plot_scores_dict(scores_dict=scores_dict, save_path=save_path, title=title)

    return scores_dict
