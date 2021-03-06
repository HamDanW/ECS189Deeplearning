'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD


# Donald Chan:
# IMPORTANT: COMMENT AND UNCOMMENT CODE DEPENDING ON WHETHER YOU ARE USING CUDA OR NOT
# IF YOU DON'T KNOW WHAT CUDA IS, COMMENT ALL CUDA LINES OUT AND UNCOMMENT THE NON-CUDA LINES.

from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_2_code.Evaluate_Precision import Evaluate_Precision
from code.stage_2_code.Evaluate_Recall import Evaluate_Recall
from code.stage_2_code.Evaluate_F1 import Evaluate_F1
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 100
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-5

    # it defines the MLP model architecture, e.g., how many layers, size of variables in each layer, activation
    # function, etc. the size of the input/output portal of the model architecture should be consistent with our data
    # input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

        # Donald Chan: Adjusted layer to account for training set. 784 features per row
        # CUDA Version
        # self.fc_layer_1 = nn.Linear(784, 784).cuda()
        # Non CUDA version
        initial = 784
        self.fc_layer_1 = nn.Linear(initial, initial)

        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html

        # CUDA version of code
        # self.activation_func_1 = nn.ReLU().cuda()
        # Non CUDA version
        self.activation_func_1 = nn.ReLU()

        # Russell Chien: hidden layer(s) implementation, haven't tried CUDA
        # self.hidden_layer_1 = nn.Linear(200, 100)
        # self.activation_hidden_layer_1 = nn.ReLU()
        factor = 2
        next_val = int(initial / factor)
        for i in range(1):
            self.hidden_layer = nn.Linear(initial, next_val)
            self.activation_hidden_layer = nn.ReLU()
            next_val = int(initial / factor)

        # Donald Chan: Layer 2 should output nx2 matrix, so it can be used in softmax function. n is the size of the input
        # CUDA version of code
        # self.fc_layer_2 = nn.Linear(784, 10).cuda()
        # Non CUDA version of code
        self.fc_layer_2 = nn.Linear(next_val, 10)

        # CUDA version
        # self.activation_func_2 = nn.Softmax(dim=1).cuda()
        # Non CUDA version
        self.activation_func_2 = nn.Softmax(dim=1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings

        # CUDA version of code
        # h = self.activation_func_1(self.fc_layer_1(x)).cuda()
        # Non CUDA version of code
        print(x.shape)
        h = self.activation_func_1(self.fc_layer_1(x))

        # output layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance

        # Russell Chien: hidden layer(s) implementation, haven't tried CUDA
        # hl_1 = self.activation_hidden_layer_1(self.hidden_layer_1(h))
        hl = self.activation_hidden_layer(self.hidden_layer(h))

        # CUDA version
        # y_pred = self.activation_func_2(self.fc_layer_2(h)).cuda()
        # Non CUDA version
        # y_pred = self.activation_func_2(self.fc_layer_2(hl_1))
        y_pred = self.activation_func_2(self.fc_layer_2(hl))

        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        # Create evaluation objects that represent evaluation metrics
        accuracy_evaluator = Evaluate_Accuracy('accuracy training evaluator', '')
        precision_evaluator = Evaluate_Precision('precision (micro) training evaluator', '')
        recall_evaluator = Evaluate_Recall('recall training evaluator', '')
        f1_evaluator = Evaluate_F1('f1 (micro) training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()

            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it

            # Convert tensors to CUDA for faster computation
            # CUDA version
            # y_pred = self.forward(torch.FloatTensor(np.array(X)).cuda()).cuda()
            # y_true = torch.LongTensor(np.array(y)).cuda()

            # Non CUDA version
            y_pred = self.forward(torch.FloatTensor(np.array(X)))

            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))

            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            optimizer.zero_grad()
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            # for CUDA, comment out if not using CUDA
            # Convert tensors to CPU for numpy functions
            # y_pred = y_pred.cpu()
            # y_true = y_true.cpu()

            if epoch % 100 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                precision_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                recall_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                f1_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item(),
                      'Precision: ', precision_evaluator.evaluate(), 'Recall: ', recall_evaluator.evaluate(),
                      'F1 (Micro): ', f1_evaluator.evaluate())

    def test(self, X):
        # do the testing, and result the result

        # CUDA Version
        # y_pred = self.forward(torch.FloatTensor(np.array(X)).cuda()).cuda()
        # Non CUDA version
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        # CUDA Version
        # pred_y = self.test(self.data['test']['X']).cuda()
        # Non CUDA Version
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}