from code.base_class.dataset import dataset

from string import punctuation

from pathlib import Path


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        if self.dataset_source_file_name == 'text_classification/':
            # set up the file paths

            train_pos_reviews = Path(
                self.dataset_source_folder_path / self.dataset_source_file_name / 'train/' / 'pos').glob('*')
            train_neg_reviews = Path(
                self.dataset_source_folder_path / self.dataset_source_file_name / 'train/' / 'neg').glob('*')
            # test_pos_reviews = Path(
            #    self.dataset_source_folder_path / self.dataset_source_file_name / 'test/' / 'pos').glob('*')
            # test_neg_reviews = Path(
            #    self.dataset_source_folder_path / self.dataset_source_file_name / 'test/' / 'neg').glob('*')

            train_text_pos = ''
            train_text_neg = ''
            # read in the reviews into one string
            for file in train_pos_reviews:
                train_file = open(file, 'r')
                train_text_pos = train_file.read()
                train_file.close()
            for file in train_neg_reviews:
                train_file = open(file, 'r')
                train_text_neg += train_file.read()
                train_file.close()
            print('loaded data')
            print(train_text_pos)

            print('cleaning data...')
            # clean the text
            train_text_pos.lower()
            train_text_neg.lower()
            train_text_pos = "".join([c for c in train_text_pos if c not in punctuation])
            train_text_neg = "".join([c for c in train_text_neg if c not in punctuation])
            all_reviews = []
            all_reviews += train_text_pos.split("\n")
            all_reviews += train_text_neg.split("\n")
            train_text_pos = " ".join(train_text_pos)
            train_text_neg = " ".join(train_text_neg)
            all_words = []
            all_words += train_text_pos.split()
            all_words += train_text_neg.split()
            print('cleaned data')

            return all_reviews, all_words
