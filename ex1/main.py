from typing import Optional

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from tqdm import tqdm

from ex1.config import synthetic_dims
from ex1.datasets.libsvm_datasets import get_diabetes_data_and_labels, get_breast_cancer_data_and_labels
from ex1.datasets.shuffle import shuffle_dataset
from ex1.datasets.synthetic_data_generation import create_dataset_by_dimention, \
    create_poor_performance_dataset_for_simple
from ex1.scripts.plot_utils import plot_mean_result_of_both_algorithms, plot_all_results_for_dataset
from ex1.scripts.rand import perform_rand_algorithm, perform_rand_algorithm_loo
from ex1.scripts.simple import perform_simple_algorithm, perform_simple_algorithm_loo

save_path = None

synthetic_loader_list = [lambda: create_dataset_by_dimention(dim) for dim in synthetic_dims]
synthetic_names_list = [f'synthetic dataset dim {dim}' for dim in synthetic_dims]

loaders_list = synthetic_loader_list + [get_diabetes_data_and_labels, get_breast_cancer_data_and_labels,
                                        create_poor_performance_dataset_for_simple]

dataset_names_list = synthetic_names_list + ['diabetes', 'breast_cancer', 'hard']

for loader_func, dataset_name in zip(loaders_list, dataset_names_list):
    x, y = loader_func()

    x_train_and_val, x_test, y_train_and_val, y_test = train_test_split(x, y, test_size=150)
    x_train, x_val, y_train, y_val = x_train_and_val[:300], x_train_and_val[300:450], y_train_and_val[
                                                                                      :300], y_train_and_val[
                                                                                             300:450]

    cv_iterable = [(list(range(300)), list(range(300, min(len(y_train_and_val), 450))))]

    linear_cv = GridSearchCV(SVC(), param_grid={'C': [0.1, 1, 10, 100], 'kernel': ['linear']}, cv=cv_iterable)
    linear_cv.fit(x_train_and_val, y_train_and_val)

    linear_params = linear_cv.best_params_

    rbf_cv = GridSearchCV(SVC(),
                          param_grid={'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10, 100], 'kernel': ['rbf']},
                          cv=cv_iterable)
    rbf_cv.fit(x_train_and_val, y_train_and_val)

    rbf_params = rbf_cv.best_params_

    # section 2

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

    # plot_mean_result_of_both_algorithms(rand_scores_dicts_list=rand_scores_dict_list,
    #                                     simple_scores_dicts_list=simple_scores_dict_list,
    #                                     title=f'{dataset_name} dataset', save_path=save_path)

    # section 3
    x_train, y_train = shuffle_dataset(x_train, y_train)
    simple_scores_dict_loo = perform_simple_algorithm_loo(x_train=x_train, y_train=y_train, linear_params=linear_params,
                                                          rbf_params=rbf_params,
                                                          title=f'simple algorithm loo {dataset_name} dataset')
    rand_scores_dict_loo = perform_rand_algorithm_loo(x_train=x_train, y_train=y_train, linear_params=linear_params,
                                                      rbf_params=rbf_params,
                                                      title=f'rand algorithm loo {dataset_name} dataset')

    plot_all_results_for_dataset(rand_scores_dicts_list=rand_scores_dict_list,
                                 simple_scores_dicts_list=simple_scores_dict_list,
                                 rand_scores_dict_loo=rand_scores_dict_loo,
                                 simple_scores_dict_loo=simple_scores_dict_loo,
                                 title=f'results for {dataset_name}',
                                 save_path=f'/home/tomk42/PycharmProjects/ee048843/ex1/outputs/{dataset_name}.png')
