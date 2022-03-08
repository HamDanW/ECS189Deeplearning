from code.base_class.method import method
import torch
from torch import nn as nn
from torch.nn import functional as F
import torch_geometric.nn as geo
import numpy as np

from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_5_code.Evaluate_Precision import Evaluate_Precision
from code.stage_5_code.Evaluate_Recall import Evaluate_Recall
from code.stage_5_code.Evaluate_F1 import Evaluate_F1

class Method_GCN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 200
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    dataset_name = ''

    # CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #From Pytorch Example
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.gcn1 = ''
        self.gcn2 = ''
        self.relu = nn.ReLU().to(self.device)
        self.dropout = nn.Dropout(p=0.2).to(self.device)
        self.log_soft = nn.LogSoftmax(dim=1).to(self.device)

    #From Pytorch Example
    def forward(self,x, edge):
        out1 = self.dropout(self.relu(self.gcn1(x, edge))).to(self.device)
        out2 = self.gcn2(out1, edge).to(self.device)
        return self.log_soft(out2).to(self.device)

    def train(self, X, y):
        # X is node feature matrix
        # y has form: [label1, label2, ..., label n]

        #Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        loss_function = nn.NLLLoss()

        

        for epoch in range(self.max_epoch + 1):  # you can do an early stop if self.max_epoch is too much...
            #Convert X and y to tensors so pytorch can operate on it
            tensorX = torch.FloatTensor(np.array(X)).to(self.device)
            # Create tensor of randomized indicies with length = length of X
            permutation = torch.randperm(tensorX.size()[0]).to(self.device)
            permuteX = tensorX[permutation].to(self.device)

            #print('tensorX: ' + str(tensorX[0]))
            #print('permuteX: ' + str(permuteX[0]))

            tensorY = torch.LongTensor(np.array(y)).to(self.device)
            permuteY = tensorY[permutation].to(self.device)

            #print('tensorY.shape: ' + str(tensorY.shape))
            #print('permuteY.shape: ' + str(permuteY.shape))

            edge_tensor = torch.LongTensor(np.array(self.data['graph']['edge'])).to(self.device)
            #Reshape edge_tensor to shape ([2, num_edges])
            edge_tensor = torch.permute(edge_tensor, (1,0)).to(self.device)
            
            #Info
            #X.shape = ([2708,1433])
            #y.shape = ([2708])
            #edge_tensor.shape = ([2,5429])

            optimizer.zero_grad()

            #y_pred = self.forward(tensorX, edge_tensor).to(self.device)
            y_pred = self.forward(permuteX, edge_tensor).to(self.device)
            y_true = permuteY.to(self.device)
            train_loss = loss_function(y_pred[self.data['train_test_val']['idx_train']], y_true[self.data['train_test_val']['idx_train']]).to(self.device)
            train_loss.backward()
            optimizer.step()


            '''
            permutation = torch.randperm(tensorX.size()[0]).to(self.device)

            pred = torch.empty(0).to(self.device)
            true = torch.empty(0).to(self.device)

            for i in range(0, tensorX.size()[0], self.batch_size):
                indicies = permutation[i:i+self.batch_size]
                miniX, miniy = tensorX[indicies], tensorY[indicies]

                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad()

                y_pred = self.forward(miniX, edge_tensor).to(self.device)
                # convert y to torch.tensor as well
                y_true = miniy

                #y_pred = y_pred - 1
                #print(y_pred[0])
                #y_true = y_true - 1


                # calculate the training loss
                train_loss = loss_function(y_pred, y_true).to(self.device)

                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                
                train_loss.backward()
                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()

                # Keep track of pred and true y values to calulate accuracy later
                pred = torch.cat((pred, y_pred), 0)
                true = torch.cat((true, y_true), 0)
            '''

            # Create evaluation objects that represent evaluation metrics
            accuracy_evaluator = Evaluate_Accuracy('accuracy training evaluator', '')
            precision_evaluator = Evaluate_Precision('precision (micro) training evaluator', '')
            recall_evaluator = Evaluate_Recall('recall training evaluator', '')
            f1_evaluator = Evaluate_F1('f1 (micro) training evaluator', '')

            if epoch % 10 == 0:
                accuracy_evaluator.data = {'true_y': y_true.to('cpu'), 'pred_y': y_pred.to('cpu').max(1)[1]}
                precision_evaluator.data = {'true_y': y_true.to('cpu'), 'pred_y': y_pred.to('cpu').max(1)[1]}
                recall_evaluator.data = {'true_y': y_true.to('cpu'), 'pred_y': y_pred.to('cpu').max(1)[1]}
                f1_evaluator.data ={'true_y': y_true.to('cpu'), 'pred_y': y_pred.to('cpu').max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item(),
                      'Precision: ', precision_evaluator.evaluate(), 'Recall: ', recall_evaluator.evaluate(),
                      'F1 (Micro): ', f1_evaluator.evaluate())


    def test(self, X):
        # do the testing, and result the result
        tensorX = torch.FloatTensor(np.array(X)).to(self.device)
        permutation = torch.randperm(tensorX.size()[0]).to(self.device)
        permuteX = tensorX[permutation].to(self.device)
        permuteX = permuteX[self.data['train_test_val']['idx_test']].to(self.device)
        print(permuteX.shape)
        edge_tensor = torch.LongTensor(np.array(self.data['graph']['edge'])).to(self.device)
        #Reshape edge_tensor to shape ([2, num_edges])
        edge_tensor = torch.permute(edge_tensor, (1,0)).to(self.device)
        print('edge tensor' + str(edge_tensor.shape))
        #y_pred = self.forward(tensorX, edge_tensor).to(self.device)
        y_pred = self.forward(permuteX, edge_tensor).to(self.device)
        test_res = y_pred.argmax(dim=1)[self.data['train_test_val']['idx_test']]

        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return test_res, permutation
    def run(self):
        #data has form:
        # {'train': {'X': trainX, 'y': trainy}, 'test': {'X': testX, 'y': testy}}
        print('method running...')
        if self.dataset_name == 'cora':
            print('cora')
            self.max_epoch = 100
            #Cora has 1433 features per line and 7 classes
            self.gcn1 = geo.GCNConv(1433,20).to(self.device)
            self.gcn2 = geo.GCNConv(20, 7).to(self.device)
        elif self.dataset_name == 'citeseer':
            self.max_epoch = 100
            #Citeseer has 1433 features per line and 6 classes
            self.gcn1 = geo.GCNConv(3703,20).to(self.device)
            self.gcn2 = geo.GCNConv(20, 6).to(self.device)
        elif self.dataset_name == 'pubmed':
            self.max_epoch = 100
            #Pubmed has 500 features per line and 3 classes
            self.gcn1 = geo.GCNConv(500,20).to(self.device)
            self.gcn2 = geo.GCNConv(20, 3).to(self.device)
        print('--start training...')
        self.train(self.data['graph']['X'], self.data['graph']['y'])
        print('--start testing...')
        pred_y, permutation = self.test(self.data['graph']['X'])
        true_y = self.data['graph']['y'][permutation]
        #print(pred_y.shape)
        pred_y = pred_y.to('cpu')
        true_y = true_y.to('cpu')
        return {'pred_y': pred_y, 'true_y': true_y}