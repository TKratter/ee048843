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
        for n in query_num_list:
            current_x_train, current_y_train = x_train[:n], y_train[:n]
            clf = SVC(kernel=kernel)
            clf.fit(current_x_train, current_y_train)

            y_pred = clf.predict(x_test)
            scores_dict[kernel].append(1 - accuracy_score(y_test, y_pred))

        plt.plot(query_num_list, scores_dict[kernel], label=kernel)
    plt.xlabel('number of queries')
    plt.ylabel('test error')
    plt.legend(loc='upper right')

    if not(title is None):
        plt.title(title)

    if save_path is None:
        plt.show()

    else:
        plt.savefig(save_path)

