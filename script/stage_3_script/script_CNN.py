from code.stage_3_code.Dataset_Loader import Dataset_Loader

import numpy as np
import torch

from pathlib import Path

data_folder_path = 'data/stage_3_data/'

CIFAR_obj = Dataset_Loader('train', '')
CIFAR_obj.dataset_source_folder_path = Path(data_folder_path)
CIFAR_obj.dataset_source_file_name = 'CIFAR'

MNIST_obj = Dataset_Loader('train', '')
MNIST_obj.dataset_source_folder_path = Path(data_folder_path)
MNIST_obj.dataset_source_file_name = 'MNIST'

ORL_obj = Dataset_Loader('train', '')
ORL_obj.dataset_source_folder_path = Path(data_folder_path)
ORL_obj.dataset_source_file_name = 'ORL'

# loads datasets 
CIFAR_obj.load()
MNIST_obj.load()
ORL_obj.load()

