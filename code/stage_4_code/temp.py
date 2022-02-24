from code.base_class.dataset import dataset

from string import punctuation

from pathlib import Path

import torch
from torchtext.data.utils import get_tokenizer

class Temp(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        if self.dataset_source_file_name == 'text_classification/':
            #set up the file paths

            '''
            train_pos_reviews = Path(
                self.dataset_source_folder_path / self.dataset_source_file_name / 'train/' / 'pos').glob('*')
            train_neg_reviews = Path(
                self.dataset_source_folder_path / self.dataset_source_file_name / 'train/' / 'neg').glob('*')

            test_pos_reviews = Path(
                self.dataset_source_folder_path / self.dataset_source_file_name / 'test/' / 'pos').glob('*')
            test_neg_reviews = Path(
                self.dataset_source_folder_path / self.dataset_source_file_name / 'test/' / 'neg').glob('*')
            '''
            test_file_path = Path('script/stage_4_script/test.txt')
            test_file = open(test_file_path, 'r', encoding='utf-8')

            reviews = []
            for line in test_file:
                reviews.append(line)
            
