import numpy as np
import torch
from code.stage_5_code import Dataset_Loader_Node_Classification

from pathlib import Path

from code.stage_5_code.Method_GCN import Method_GCN
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_5_code.Evaluate_Precision import Evaluate_Precision
from code.stage_5_code.Evaluate_Recall import Evaluate_Recall
from code.stage_5_code.Evaluate_F1 import Evaluate_F1


np.random.seed(2)
torch.manual_seed(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_folder_path = 'data/stage_5_data/'
data_file_name = 'cora'

dataloader = Dataset_Loader_Node_Classification.Dataset_Loader()
dataloader.dataset_source_folder_path = data_folder_path + data_file_name
dataloader.dataset_name = data_file_name

#data has form {'graph':{'node': {}, 'edge':{}, 'X': {}, 'y':{}, 'utility':{'A': {}, 'reverse_idx':{}}}, 'train_test_val':{'idx_train':{}, 'idx_test':{}, 'idx_val': {}}}
data = dataloader.load()

method_obj = Method_GCN('GCN','')
method_obj.data = data
method_obj.dataset_name = data_file_name
test_results = method_obj.run()

# Prep results for stat
predictions = test_results['pred_y']
expected = test_results['true_y']


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

print('Done')
