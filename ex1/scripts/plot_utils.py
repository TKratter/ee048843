import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np

from ex1.config import query_num_list


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
    unified_scores_dict = defaultdict(list)
    for scores_dict in scores_dicts_list:
        for kernel, scores_list in scores_dict.items():
            unified_scores_dict[kernel].append(scores_list)

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
