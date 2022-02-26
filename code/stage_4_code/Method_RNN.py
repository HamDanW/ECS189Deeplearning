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

    max_epoch = 100
    learning_rate = .1
    

    # CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.embedding = ''
        self.LSTM = ''
        self.fc1 = nn.Linear(3, 10).to(self.device)
        self.drop = nn.Dropout(p=0.2).to(self.device)
        self.soft = nn.Softmax(dim=1)


    def forward(self, x):
        embed = self.embedding(x).to(self.device)
        dropped = self.drop(embed).to(self.device)
        out, (h_state, c_state) = self.LSTM(dropped).to(self.device)
        fc = self.fc1(h_state[-1]).to(self.device)
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
            tensorX = torch.FloatTensor(np.array(X, dtype='int64')).to(self.device)
            tensorY = torch.FloatTensor(np.array(y)).to(self.device)
            y_pred = self.forward(tensorX).to(self.device)
            y_true = tensorY

            # calculate the training loss
            train_loss = loss_function(y_pred, y_true).to(self.device)

            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
                
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

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

    '''
    def test(self, X):
        # TODO
    '''

    def run(self):
        print('method running...')
        #Assume input is a list of encoded sentences
        #Initalize Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=len(self.data['all_words']), embedding_dim=7, padding_idx=0).to(self.device)
        self.LSTM = nn.LSTM(input_size=7, hidden_size=3, num_layers=3, batch_first=True).to(self.device)
        trainX = self.data['train']['X']
        #print('Type: ' + str(np.array(trainX).dtype))
        trainY = self.data['train']['y']
        self.train(trainX, trainY)

