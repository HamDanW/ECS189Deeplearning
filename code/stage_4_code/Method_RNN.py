from code.base_class.method import method
import torch
from torch import index_put_, nn as nn
import torch.nn.functional as F
import numpy as np

from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_4_code.Evaluate_Precision import Evaluate_Precision
from code.stage_4_code.Evaluate_Recall import Evaluate_Recall
from code.stage_4_code.Evaluate_F1 import Evaluate_F1


class Method_RNN(method, nn.Module):
    data = None

    text_class = False
    max_epoch = 200
    learning_rate = 1e-3
    num_layers = 3
    num_hidden = 200
    batch_size = 500

    #text gen
    vocab_size = 0
    

    # CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.embedding = ''
        self.LSTM = ''
        self.fc1 = ''
        self.drop = nn.Dropout(p=0.2).to(self.device)
        self.soft = nn.Softmax(dim=1)


    def forward(self, x):
        embed = self.embedding(x).to(self.device)
        dropped = self.drop(embed).to(self.device)
        out, (hidden, cell) = self.LSTM(dropped)
        if not self.text_class:
            out = out.reshape(-1, self.num_hidden)
            fc = self.fc1(self.drop(out)).to(self.device)
        elif self.text_class:
            fc = self.fc1(self.drop(hidden[-1])).to(self.device)
        soft = self.soft(fc)
        return soft

    def train(self, X, y):
        #Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        
        loss_function = nn.CrossEntropyLoss()

        for epoch in range(0, self.max_epoch+1):
            
            #Assume X is review
            optimizer.zero_grad()
            
            #Convert input to tensor
            tensorX = torch.LongTensor(np.array(X)).to(self.device)
            tensorY = torch.LongTensor(np.array(y)).to(self.device)
            permutation = torch.randperm(tensorX.size()[0]).to(self.device)

            for i in range(0, tensorX.size()[0], self.batch_size):
                indicies = permutation[i:i+self.batch_size].to(self.device)
                #miniX, miniy = torch.LongTensor(tensorX[indicies].to(self.device)).to(self.device), torch.LongTensor(tensorY[indicies].to(self.device)).to(self.device)
                miniX, miniy = tensorX[indicies], tensorY[indicies]

                y_pred = self.forward(miniX).to(self.device)
                y_true = miniy

                # calculate the training loss
                if not self.text_class:
                    train_loss = loss_function(y_pred, y_true.view(-1)).to(self.device)
                elif self.text_class:
                    train_loss = loss_function(y_pred, y_true).to(self.device)
                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                    
                train_loss.backward()
                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()

            if self.text_class:
                # Create evaluation objects that represent evaluation metrics
                accuracy_evaluator = Evaluate_Accuracy('accuracy training evaluator', '')
                precision_evaluator = Evaluate_Precision('precision (micro) training evaluator', '')
                recall_evaluator = Evaluate_Recall('recall training evaluator', '')
                f1_evaluator = Evaluate_F1('f1 (micro) training evaluator', '')

                if epoch % 10 == 0:
                    print(y_pred.shape)
                    print(y_true.shape)
                    accuracy_evaluator.data = {'true_y': y_true.to('cpu'), 'pred_y': y_pred.to('cpu').max(1)[1]}
                    precision_evaluator.data = {'true_y': y_true.to('cpu'), 'pred_y': y_pred.to('cpu').max(1)[1]}
                    recall_evaluator.data = {'true_y': y_true.to('cpu'), 'pred_y': y_pred.to('cpu').max(1)[1]}
                    f1_evaluator.data ={'true_y': y_true.to('cpu'), 'pred_y': y_pred.to('cpu').max(1)[1]}
                    print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item(),
                        'Precision: ', precision_evaluator.evaluate(), 'Recall: ', recall_evaluator.evaluate(),
                        'F1 (Micro): ', f1_evaluator.evaluate())

    
    def test(self, X, y):
        '''
        tensor = torch.LongTensor(X).to(self.device)
        y_pred = self.forward(tensor).to(self.device)
        '''
        pred_results = torch.empty(0).to(self.device)
        true_results = torch.empty(0).to(self.device)
        tensorX = torch.LongTensor(X).to(self.device)
        tensory = torch.LongTensor(y).to(self.device)
        permutation = torch.randperm(tensorX.size()[0]).to(self.device)
        for i in range(0, tensorX.size()[0], self.batch_size):
            indicies = permutation[i:i+self.batch_size].to(self.device)
            miniX = tensorX[indicies]
            miniy = tensory[indicies]
            y_pred = self.forward(miniX).to(self.device)
            true_results = torch.cat((true_results, miniy),0)
            pred_results = torch.cat((pred_results, y_pred.max(1)[1]), 0)

        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return {'pred_y': pred_results.to('cpu'), 'true_y': true_results.to('cpu')}
    

    def run(self):
        print('method running...')
        #Text Classification Enabled
        if self.text_class:
            #Assume input is a list of encoded sentences
            #Initalize Embedding Layer
            self.embedding = nn.Embedding(num_embeddings=len(self.data['all_words']), embedding_dim=300, padding_idx=0).to(self.device)
            self.LSTM = nn.LSTM(input_size=300, hidden_size=self.num_hidden, num_layers=self.num_layers, batch_first=True).to(self.device)
            self.fc1 = nn.Linear(200, 2).to(self.device)
            print('-----------------Start Training-----------------')
            trainX = self.data['train']['X']
            #print('Type: ' + str(np.array(trainX).dtype))
            trainY = self.data['train']['y']
            self.train(trainX, trainY)
            print('-----------------Training Done-----------------')
            print('-----------------Start Testing-----------------')
            results = self.test(self.data['test']['X'], self.data['test']['y'])
            print('-----------------Testing Done-----------------')
            return results
        elif not self.text_class:
            print('Text Gen')
            vocab_size = len(self.data['all_words'])
            self.num_hidden = 300
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=300, padding_idx=0).to(self.device)
            self.LSTM = nn.LSTM(input_size=300, hidden_size=self.num_hidden, num_layers=self.num_layers, batch_first=True).to(self.device)
            self.fc1 = nn.Linear(300, vocab_size).to(self.device)
            
            print('-----------------Start Training-----------------')
            trainX = self.data['X']
            #print('Type: ' + str(np.array(trainX).dtype))
            trainY = self.data['y']
            self.train(trainX, trainY)
            print('-----------------Training Done-----------------')

