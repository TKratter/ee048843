import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np

from ex1.config import query_num_list

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['font.size'] = 16


def plot_all_results_for_dataset(rand_scores_dicts_list: List[Dict[str, List[float]]],
                                 simple_scores_dicts_list: List[Dict[str, List[float]]],
                                 rand_scores_dict_loo: Dict[str, List[float]],
                                 simple_scores_dict_loo: Dict[str, List[float]],
                                 save_path: Optional[str] = None,
                                 title: Optional[str] = None):
    plt.figure()
    for kernel in rand_scores_dict_loo.keys():
        plt.plot(query_num_list, rand_scores_dict_loo[kernel], label=f'kernel: {kernel}, algorithm: rand LOO')

    for kernel in simple_scores_dict_loo.keys():
        plt.plot(query_num_list, simple_scores_dict_loo[kernel],
                 label=f'kernel: {kernel}, algorithm: simple LOO')

    rand_unified_scores_dict = _get_unified_scores_dict(rand_scores_dicts_list)
    simple_unified_scores_dict = _get_unified_scores_dict(simple_scores_dicts_list)

    for kernel, list_of_scores_list in rand_unified_scores_dict.items():
        plt.plot(query_num_list, np.array(list_of_scores_list).mean(axis=0),
                 label=f'kernel: {kernel}, algorithm: rand mean over 30')

    for kernel, list_of_scores_list in simple_unified_scores_dict.items():
        plt.plot(query_num_list, np.array(list_of_scores_list).mean(axis=0),
                 label=f'kernel: {kernel}, algorithm: simple mean over 30')

    plt.xlabel('number of queries')
    plt.ylabel('mean test error')
    plt.legend(loc='upper right', prop={'size': 8})

    if not (title is None):
        plt.title(title)

    if save_path is None:
        plt.show()

    else:
        plt.savefig(save_path)


def plot_scores_dict(scores_dict: Dict[str, List[float]], save_path: Optional[str] = None,
                     title: Optional[str] = None):
    plt.figure()
    for kernel in scores_dict.keys():
        plt.plot(query_num_list, scores_dict[kernel], label=kernel)

    plt.xlabel('number of queries')
    plt.ylabel('test error')
    plt.legend(loc='upper right')

    if not (title is None):
        plt.title(title)

    if save_path is None:
        plt.show()

    else:
        plt.savefig(save_path)


def plot_mean_result_of_scores_dicts_list(scores_dicts_list: List[Dict[str, List[float]]],
                                          save_path: Optional[str] = None,
                                          title: Optional[str] = None):
    plt.figure()
    unified_scores_dict = _get_unified_scores_dict(scores_dicts_list)

    for kernel, list_of_scores_list in unified_scores_dict.items():
        plt.plot(query_num_list, np.array(list_of_scores_list).mean(axis=0), label=kernel)

    plt.xlabel('number of queries')
    plt.ylabel('mean test error')
    plt.legend(loc='upper right')

    if not (title is None):
        plt.title(title)

    if save_path is None:
        plt.show()

    else:
        plt.savefig(save_path)


def plot_mean_result_of_both_algorithms(rand_scores_dicts_list: List[Dict[str, List[float]]],
                                        simple_scores_dicts_list: List[Dict[str, List[float]]],
                                        save_path: Optional[str] = None,
                                        title: Optional[str] = None):
    rand_unified_scores_dict = _get_unified_scores_dict(rand_scores_dicts_list)
    simple_unified_scores_dict = _get_unified_scores_dict(simple_scores_dicts_list)

    plt.figure()
    for kernel, list_of_scores_list in rand_unified_scores_dict.items():
        plt.plot(query_num_list, np.array(list_of_scores_list).mean(axis=0), label=f'kernel: {kernel}, algorithm: rand')

    for kernel, list_of_scores_list in simple_unified_scores_dict.items():
        plt.plot(query_num_list, np.array(list_of_scores_list).mean(axis=0),
                 label=f'kernel: {kernel}, algorithm: simple')

    plt.xlabel('number of queries')
    plt.ylabel('mean test error')
    plt.legend(loc='upper right')

    if not (title is None):
        plt.title(title)

    if save_path is None:
        plt.show()

    else:
        plt.savefig(save_path)


def _get_unified_scores_dict(scores_dicts_list: List[Dict[str, List[float]]]) -> Dict[str, List[List[float]]]:
    unified_scores_dict = defaultdict(list)
    for scores_dict in scores_dicts_list:
        for kernel, scores_list in scores_dict.items():
            unified_scores_dict[kernel].append(scores_list)
    return unified_scores_dict
