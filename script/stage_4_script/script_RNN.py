from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Method_RNN import Method_RNN
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_4_code.Evaluate_Precision import Evaluate_Precision
from code.stage_4_code.Evaluate_Recall import Evaluate_Recall
from code.stage_4_code.Evaluate_F1 import Evaluate_F1
from code.stage_3_code.Result_Saver import Result_Saver


import numpy as np
import torch

from pathlib import Path


np.random.seed(2)
torch.manual_seed(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# change data_file_name for text_classifcation/text_generation
data_folder_path = 'data/stage_4_data/'
data_file_name = 'text_generation/'

if data_file_name == 'text_classification/':
    data_obj = Dataset_Loader('train', '')
    data_obj.dataset_source_folder_path = Path(data_folder_path)
    data_obj.dataset_source_file_name = data_file_name

    #train_all_reviews, train_all_words, test_all_reviews, test_all_words = data_obj.load()
    train_data, train_y, test_data, test_y, vocab = data_obj.load()

    input = {'train': {'X': train_data, 'y': train_y}, 'test': {'X': test_data, 'y': test_y}, 'all_words': vocab}

    model = Method_RNN('LSTM', '')
    model.data = input
    model.text_class = True
    model.max_epoch = 200
    model.to(device)
    test_results = model.run()

    # Prep results for stat
    predictions = test_results['pred_y']
    expected = test_results['true_y']

    #print(predictions)
    #print(expected)

    # Output Stat Measurements
    accuracy_evaluator = Evaluate_Accuracy('accuracy training evaluator', '')
    precision_evaluator = Evaluate_Precision('precision (micro) training evaluator', '')
    recall_evaluator = Evaluate_Recall('recall training evaluator', '')
    f1_evaluator = Evaluate_F1('f1 (micro) training evaluator', '')

    accuracy_evaluator.data = {'true_y': expected, 'pred_y': predictions}
    precision_evaluator.data = {'true_y': expected, 'pred_y': predictions}
    recall_evaluator.data = {'true_y': expected, 'pred_y': predictions}
    f1_evaluator.data = {'true_y': expected, 'pred_y': predictions}

    print('Overall Accuracy: ' + str(accuracy_evaluator.evaluate()))
    print('Overall Precision: ' + str(precision_evaluator.evaluate()))
    print('Overall Recall: ' + str(recall_evaluator.evaluate()))
    print('Overall F1: ' + str(f1_evaluator.evaluate()))

    # Russell Chien: Put your folder path here for results
    result_folder_path = Path('result/stage_3_result/')
    result_folder_name = 'CNN_prediction_result'

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = result_folder_path
    result_obj.result_destination_file_name = result_folder_name


    print('Done')
elif data_file_name == 'text_generation/':
    data_obj = Dataset_Loader('train', '')
    data_obj.dataset_source_folder_path = Path(data_folder_path)
    data_obj.dataset_source_file_name = data_file_name

    #Load data
    encoded_x, encoded_y, sequences, vocab, lookup = data_obj.load()

    input = {'X': encoded_x, 'y': encoded_y, 'word2int': vocab, 'sequences': sequences, 'int2word': lookup}

    #Create Model
    model = Method_RNN('LSTM', '')
    model.data = input
    model.text_class = False
    model.max_epoch = 1000
    model.to(device)
    model.run()


    print('Done')