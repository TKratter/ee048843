from typing import Optional

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from tqdm import tqdm

from ex1.datasets.libsvm_datasets import get_diabetes_data_and_labels, get_breast_cancer_data_and_labels
from ex1.datasets.shuffle import shuffle_dataset
from ex1.scripts.plot_utils import plot_mean_result_of_both_algorithms
from ex1.scripts.rand import perform_rand_algorithm
from ex1.scripts.simple import perform_simple_algorithm


def section_2(save_path: Optional[str] = None):
    for loader_func, dataset_name in zip([get_diabetes_data_and_labels, get_breast_cancer_data_and_labels],
                                         ['diabetes', 'breast_cancer']):
        x, y = loader_func()

        x_train_and_val, x_test, y_train_and_val, y_test = train_test_split(x, y, test_size=150)
        x_train, x_val, y_train, y_val = x_train_and_val[:300], x_train_and_val[300:450], y_train_and_val[
                                                                                          :300], y_train_and_val[
                                                                                                 300:450]

        cv_iterable = [(list(range(300)), list(range(300, min(len(y_train_and_val), 450))))]

        liner_cv = GridSearchCV(SVC(), param_grid={'C': [0.1, 1, 10, 100], 'kernel': ['linear']}, cv=cv_iterable)
        liner_cv.fit(x_train_and_val, y_train_and_val)

        linear_params = liner_cv.best_params_

        rbf_cv = GridSearchCV(SVC(),
                              param_grid={'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10, 100], 'kernel': ['rbf']},
                              cv=cv_iterable)
        rbf_cv.fit(x_train_and_val, y_train_and_val)

        rbf_params = rbf_cv.best_params_

        rand_scores_dict_list = []
        simple_scores_dict_list = []

        for i in tqdm(range(30)):
            x_train, y_train = shuffle_dataset(x_train, y_train)
            rand_scores_dict_list.append(
                perform_rand_algorithm(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                       linear_params=linear_params, rbf_params=rbf_params,
                                       title=f'rand algorithm {dataset_name} dataset'))
            simple_scores_dict_list.append(
                perform_simple_algorithm(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                         linear_params=linear_params, rbf_params=rbf_params,
                                         title=f'simple algorithm {dataset_name} dataset'))

        plot_mean_result_of_both_algorithms(rand_scores_dicts_list=rand_scores_dict_list,
                                            simple_scores_dicts_list=simple_scores_dict_list,
                                            title=f'{dataset_name} dataset', save_path=save_path)