from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_KFold_CV import Setting_KFold_CV
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

from pathlib import Path

# ---- Multi-Layer Perceptron script ----
# ---- parameter section -------------------------------
np.random.seed(2)
torch.manual_seed(2)
# ------------------------------------------------------

# ---- objection initialization section ---------------
# data_obj = Dataset_Loader('toy', '')
# data_obj.dataset_source_folder_path = 'data/stage_1_data/'
# data_obj.dataset_source_file_name = 'toy_data_file.txt'
# My code
# initialize training and test datasets

# Russell Chien: Put your folder path here for testing data
data_folder_path = 'data/stage_3_data/'
data_file_name = 'CIFAR'

data_obj = Dataset_Loader('data', '')
data_obj.dataset_source_folder_path = Path(data_folder_path)
data_obj.dataset_source_file_name = data_file_name


# Initialize CNN Object
method_obj = Method_CNN('convolution layers', '')

# Use cuda for model
# Comment out if you don't know what cuda is
# Make sure all lines with ".cuda()" are commented out in Method_MLP.py
# method_obj = method_obj.cuda()

# My code
# Load train dataset and test dataset and create a dictionary with labels 'train' and 'test'.
# 'train' and 'test' are names used in stage_1_code/Method_MLP.py
# Store dictionary into method_obj's (MLP_Method object) data variable
# data has the form: {'train': {'X': trainX, 'y': trainy}, 'test': {'X': testX, 'y': testy}}
data = data_obj.load()
method_obj.data = data
# Call run function for CNN model object
# CNN model object functions located in stage_3_code/Method_CNN.py

method_obj.run()

# Russell Chien: Put your folder path here for results
result_folder_path = Path('result/stage_3_result/')
result_folder_name = 'CNN_prediction_result'

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = result_folder_path
result_obj.result_destination_file_name = result_folder_name

setting_obj = Setting_KFold_CV('k fold cross validation', '')
# setting_obj = Setting_Tra
# in_Test_Split('train test split', '')

evaluate_obj = Evaluate_Accuracy('accuracy', '')
# ------------------------------------------------------

# ---- running section ---------------------------------
print('************ Start ************')
setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj, evaluate_obj, evaluate_obj, evaluate_obj)
setting_obj.print_setup_summary()
mean_score, std_score = setting_obj.load_run_save_evaluate()
print('************ Overall Performance ************')
print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
print('************ Finish ************')
# ------------------------------------------------------
