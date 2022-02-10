'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.stage_2_code.evaluate import evaluate
from sklearn.metrics import accuracy_score


class Evaluate_Accuracy(evaluate):
    data = None
<<<<<<< HEAD
    
=======

>>>>>>> da58fdbb6775c3eceab070565ef4b0fd2fb00b9f
    def evaluate(self):
        print('Evaluating Accuracy...')
        return accuracy_score(self.data['true_y'], self.data['pred_y'])
