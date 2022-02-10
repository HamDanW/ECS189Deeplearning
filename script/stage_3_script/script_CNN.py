from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_3_code.Dataset_Loader import Dataset_Loader

import torch
import torch.nn as nn

import numpy as np

from pathlib import Path


net = Method_CNN('CNN Model', '')

data_folder_path = 'data/stage_3_data/'

MNIST_data_obj = Dataset_Loader('train', '')
MNIST_data_obj.dataset_source_folder_path = Path(data_folder_path)
MNIST_data_obj.dataset_source_file_name = 'MNIST'

train = MNIST_data_obj.load('train')
test = MNIST_data_obj.load('test')
net.data = {'train': train, 'test': test}

net.run()
