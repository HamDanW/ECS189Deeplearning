'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

# Donald Chan : Modified for Stage 2 assignment

from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        X = []
        y = []
        file = self.dataset_source_folder_path / self.dataset_source_file_name
        f = open(file)
        for line in f:
            line = line.strip('\n')
            # elements = [int(i) for i in line.split(' ')]
            # My change
            # Let X be the features (input)
            # Let y be the label    (output)
            elements = [int(i) for i in line.split(',')]
            # X.append(elements[:-1])
            X.append(elements[1:])
            y.append(elements[0])
        f.close()
        return {'X': X, 'y': y}
