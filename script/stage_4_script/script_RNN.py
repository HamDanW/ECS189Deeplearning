from cgi import test
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
data_file_name = 'text_generation/'

data_obj = Dataset_Loader('train', '')
data_obj.dataset_source_folder_path = Path(data_folder_path)
data_obj.dataset_source_file_name = data_file_name

#train_all_reviews, train_all_words, test_all_reviews, test_all_words = data_obj.load()

if data_file_name == 'text_classification/':
    data_obj.load_from_files = True

    #train_all_reviews, train_all_words, test_all_reviews, test_all_words = data_obj.load()
    train_data, train_y, test_data, test_y, vocab = data_obj.load()

    input = {'train': {'X': train_data, 'y': train_y}, 'test': {'X': test_data, 'y': test_y}}
    print('Data loaded')

    input_size = 187096
    embed_dim = 100
    hidden_dim = 256
    output_dim = 1
    n_layers = 2
    bidirectional = True
    dropout = .5

    model = Method_RNN('RNN model', '',
                       vocab_size=input_size, embedding_dim=embed_dim,
                       hidden_dim=hidden_dim, output_dim=output_dim,
                       n_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
    model.train(input['train']['X'], input['train']['y'])

elif data_file_name == 'text_generation/':
    X, y = data_obj.load()
    print('Data loaded')

    input_size = 23415
    embed_dim = 128
    hidden_dim = 128
    output_dim = 1
    n_layers = 3
    bidirectional = True
    dropout = .2

    model = Method_RNN('RNN model', '',
                       vocab_size=input_size, embedding_dim=embed_dim,
                       hidden_dim=hidden_dim, output_dim=output_dim,
                       n_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
    model.train(X, y)
