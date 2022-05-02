import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from ex1.config import query_num_list


def plot_scores_dict(scores_dict: Dict[str, List[float]], save_path: Optional[str] = None,
                     title: Optional[str] = None):
    plt.figure()
    for kernel in scores_dict.keys():
        plt.plot(query_num_list, scores_dict[kernel], label=kernel)

    if not (title is None):
        plt.title(title)

    if save_path is None:
        plt.show()

    else:
        plt.savefig(save_path)
