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

    total_num = 0
    train_size = 0
    test_size = 0
    rand_nums = []

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
            # Randomization
            self.total_num = len(self.data['train_test_val']['idx_train']) + len(self.data['train_test_val']['idx_test'] + 1)
            self.rand_nums = torch.randperm(self.total_num).to(self.device)
            self.train_size = len(self.data['train_test_val']['idx_train'])
            self.test_size = len(self.data['train_test_val']['idx_test'])
            train_y = []
            for i in range(0,self.train_size):
                train_y.append(y[self.rand_nums[i]])

            
            #Convert X and y to tensors so pytorch can operate on it
            tensorX = torch.FloatTensor(np.array(X)).to(self.device)

            tensorY = torch.LongTensor(np.array(train_y)).to(self.device)

            edge_tensor = torch.LongTensor(np.array(self.data['graph']['edge'])).to(self.device)
            #Reshape edge_tensor to shape ([2, num_edges])
            edge_tensor = torch.permute(edge_tensor, (1,0)).to(self.device)

            optimizer.zero_grad()

            #y_pred = self.forward(tensorX, edge_tensor).to(self.device)
            y_pred = self.forward(tensorX, edge_tensor).to(self.device)
            

            #randomization
            sampled_pred = []
            for i in range(0,self.train_size):
                sampled_pred.append(y_pred[self.rand_nums[i]].to(self.device))
            sampled_pred = torch.stack(sampled_pred).to(self.device)


            y_true = tensorY.to(self.device)

            train_loss = loss_function(sampled_pred, y_true).to(self.device)
            train_loss.backward()
            optimizer.step()

            # Create evaluation objects that represent evaluation metrics
            accuracy_evaluator = Evaluate_Accuracy('accuracy training evaluator', '')
            precision_evaluator = Evaluate_Precision('precision (micro) training evaluator', '')
            recall_evaluator = Evaluate_Recall('recall training evaluator', '')
            f1_evaluator = Evaluate_F1('f1 (micro) training evaluator', '')

            if epoch % 10 == 0:
                accuracy_evaluator.data = {'true_y': y_true.to('cpu'), 'pred_y': sampled_pred.to('cpu').max(1)[1]}
                precision_evaluator.data = {'true_y': y_true.to('cpu'), 'pred_y':sampled_pred.to('cpu').max(1)[1]}
                recall_evaluator.data = {'true_y': y_true.to('cpu'), 'pred_y': sampled_pred.to('cpu').max(1)[1]}
                f1_evaluator.data ={'true_y': y_true.to('cpu'), 'pred_y': sampled_pred.to('cpu').max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item(),
                      'Precision: ', precision_evaluator.evaluate(), 'Recall: ', recall_evaluator.evaluate(),
                      'F1 (Micro): ', f1_evaluator.evaluate())


    def test(self, X):
        # do the testing, and result the result
        tensorX = torch.FloatTensor(np.array(X)).to(self.device)
        edge_tensor = torch.LongTensor(np.array(self.data['graph']['edge'])).to(self.device)
        #Reshape edge_tensor to shape ([2, num_edges])
        edge_tensor = torch.permute(edge_tensor, (1,0)).to(self.device)
        y_pred = self.forward(tensorX, edge_tensor).to(self.device)

        #randomization
        sampled_pred = []
        for i in range(self.train_size+1,self.test_size):
            sampled_pred.append(y_pred[self.rand_nums[i]].to(self.device))
        sampled_pred = torch.stack(sampled_pred).to(self.device)
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        test_res = sampled_pred.argmax(dim=1)

        true_y = []
        y = self.data['graph']['y']

        for i in range(self.train_size+1,self.test_size):
                true_y.append(y[self.rand_nums[i]])
        true_y = torch.LongTensor(np.array(true_y)).to(self.device)


        return test_res, true_y
    def run(self):
        #data has form:
        # {'train': {'X': trainX, 'y': trainy}, 'test': {'X': testX, 'y': testy}}
        print('method running...')
        if self.dataset_name == 'cora':
            print('cora')
            self.max_epoch = 1000
            #Cora has 1433 features per line and 7 classes
            self.gcn1 = geo.GCNConv(1433,20).to(self.device)
            self.gcn2 = geo.GCNConv(20, 7).to(self.device)
        elif self.dataset_name == 'citeseer':
            print('citeseer')
            self.max_epoch = 1000
            #Citeseer has 1433 features per line and 6 classes
            self.gcn1 = geo.GCNConv(3703,20).to(self.device)
            self.gcn2 = geo.GCNConv(20, 6).to(self.device)
        elif self.dataset_name == 'pubmed':
            print('pubmed')
            self.max_epoch = 1000
            #Pubmed has 500 features per line and 3 classes
            self.gcn1 = geo.GCNConv(500,20).to(self.device)
            self.gcn2 = geo.GCNConv(20, 3).to(self.device)
        print('--start training...')
        self.train(self.data['graph']['X'], self.data['graph']['y'])
        print('--start testing...')
        pred_y, true_y = self.test(self.data['graph']['X'])
        #print(pred_y.shape)
        pred_y = pred_y.to('cpu')
        true_y = true_y.to('cpu')
        return {'pred_y': pred_y, 'true_y': true_y}