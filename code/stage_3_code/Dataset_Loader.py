from code.base_class.dataset import dataset
import pickle
import matplotlib.pyplot as plt


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self, dType):
        print('loading data...')
        X = []
        y = []
        file = self.dataset_source_folder_path / self.dataset_source_file_name
        f = open(file, 'rb')
        data = pickle.load(f)

        print(dType, ' set size:', len(data[dType]))

        for pair in data[dType]:
            X.append(pair['image'])
            y.append(pair['label'])
        return {'X': X, 'y': y}
