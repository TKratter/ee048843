import numpy as np
from typing import List, Optional, Dict
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import defaultdict
import matplotlib.pyplot as plt

from ex1.config import query_num_list
from ex1.scripts.plot_utils import plot_scores_dict


def perform_rand_algorithm(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                           linear_params: dict, rbf_params: dict,
                           save_path: Optional[str] = None,
                           title: Optional[str] = None, verbose: Optional[bool] = False) -> Dict[str, List[float]]:
    scores_dict = defaultdict(list)
    for params in [linear_params, rbf_params]:
        for n in query_num_list:
            current_x_train, current_y_train = x_train[:n], y_train[:n]
            clf = SVC(**params)
            clf.fit(current_x_train, current_y_train)

            y_pred = clf.predict(x_test)
            scores_dict[params['kernel']].append(1 - accuracy_score(y_test, y_pred))

    if verbose:
        plot_scores_dict(scores_dict=scores_dict, save_path=save_path, title=title)

    return scores_dict


def perform_rand_algorithm_loo(x_train: np.ndarray, y_train: np.ndarray,
                               linear_params: dict, rbf_params: dict,
                               save_path: Optional[str] = None, title: Optional[str] = None,
                               verbose: Optional[bool] = False) -> Dict[str, List[float]]:
    scores_dict = defaultdict(list)
    for params in [linear_params, rbf_params]:
        for n in query_num_list:
            current_x_train, current_y_train = x_train[:n], y_train[:n]
            clf = SVC(**params)

            scores = cross_val_score(clf, current_x_train, current_y_train,
                                     cv=LeaveOneOut().split(current_x_train, current_y_train))
            scores_dict[params['kernel']].append(1 - scores.mean())

    if verbose:
        plot_scores_dict(scores_dict=scores_dict, save_path=save_path, title=title)

    return scores_dict
