import numpy as np
from typing import List, Optional
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import defaultdict
import matplotlib.pyplot as plt


def perform_rand_algorithm(x: np.ndarray, y: np.ndarray, query_num_list: List[int], save_path: Optional[str] = None,
                           title: Optional[str] = None):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=150)
    scores_dict = defaultdict(list)
    plt.figure()
    for kernel in ['linear', 'rbf']:
        # todo: create query_mechanism
        pass