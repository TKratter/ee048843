from ex1.config import query_num_list
from ex1.scripts.rand import perform_rand_algorithm
from ex1.scripts.simple import perform_simple_algorithm
from ex1.synthetic_data_generation import create_dataset_by_dimention
from libsvm_datasets import get_breast_cancer_data_and_labels, get_diabetes_data_and_labels

for loader_func, dataset_name in zip([get_diabetes_data_and_labels, get_breast_cancer_data_and_labels],
                                     ['diabetes', 'breast_cancer']):
    x, y = loader_func()
    perform_rand_algorithm(x=x, y=y, query_num_list=query_num_list, title=f'rand algorithm {dataset_name} dataset')
    perform_simple_algorithm(x=x, y=y, title=f'simple algorithm {dataset_name} dataset')

