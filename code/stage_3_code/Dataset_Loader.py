<<<<<<< HEAD
'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

# Donald Chan : Modified for Stage 3 assignment
import pickle

from code.base_class.dataset import dataset
=======
from code.base_class.dataset import dataset
import pickle
import matplotlib.pyplot as plt
>>>>>>> da58fdbb6775c3eceab070565ef4b0fd2fb00b9f


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

<<<<<<< HEAD
    def load(self):
        print('loading data...')
        trainX = []
        trainy = []
        testX = []
        testy = []
        finalTrain = []
        file = self.dataset_source_folder_path / self.dataset_source_file_name
        f = open(file, 'rb')
        data = pickle.load(f)
        f.close()

        # trainX = Form = [[image1][image2]...[image n]]
        # trainy = Form = [label1, label2, ..., label n]
        for element in data['train']:
            #trainX = array representation of image
            #trainy = label
            trainX.append(element['image'])
            trainy.append(element['label'])

        
        for element in data['test']:
            #testX = array representation of image
            #testy = label
            testX.append(element['image'])
            testy.append(element['label'])

        return {'train': {'X': trainX, 'y': trainy}, 'test': {'X': testX, 'y': testy}}
=======
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
>>>>>>> da58fdbb6775c3eceab070565ef4b0fd2fb00b9f
