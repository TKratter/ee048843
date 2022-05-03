import numpy as np
from sklearn.svm import SVC
from scipy.stats import multivariate_normal


def estimate_bayes_error(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                         clf_params: dict):
    dim = x_train.shape[-1]
    clf = SVC(**clf_params)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    zero_one_loss = 1 - (y_pred == y_test).astype(int)

    print(f'bayes error estimation for synthetic dataset of dimention {dim} is {zero_one_loss.mean()}')
