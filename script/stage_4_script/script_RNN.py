from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Method_RNN import Method_RNN


import numpy as np
import torch

from pathlib import Path

np.random.seed(2)
torch.manual_seed(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# change data_file_name for text_classifcation/text_generation
data_folder_path = 'data/stage_4_data/'
data_file_name = 'text_classification/'

data_obj = Dataset_Loader('data', '')
data_obj.dataset_source_folder_path = Path(data_folder_path)
data_obj.dataset_source_file_name = data_file_name

all_reviews, all_words = data_obj.load()
