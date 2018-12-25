import csv
import numpy as np
import random

########################################################
# Feed in some predictions and correct_values,
# this class calculates some important assessment measures.
# Input specification:
# correct_values[i] is taken to be the correct value for predictions[i]
class Evaluation:
    def __init__(self, predictions = [], correct_values = []):
        self.predictions = predictions
        self.correct_values = correct_values
        
        self.true_pos = 0 #pred[i] = cor[i] = 1
        self.true_neg = 0 #pred[i] = cor[i] = 0
        self.false_pos = 0 #pred[i] = 1, cor[i] = 0
        self.false_neg = 0 #pred[i] = 0, cor[i] = 1
        self.total = len(predictions)
    
        self.calculate_confusion_matrix()
    
    def accuracy(self):
        return (self.true_pos + self.true_neg) / self.total
        
    def recall(self):
        return self.true_pos / (self.true_pos + self.false_neg)
    
    def precision(self):
        return self.true_pos / (self.true_pos + self.false_pos)
    
    def F1(self):
        return 2*(self.precision()*self.recall()) / (self.precision()+self.recall())
    
    # Calculates the confusion matrix based on the objects attributes
    # self.predictions, self.correct_values
    def calculate_confusion_matrix(self):
        for i in range(len(self.predictions)):
            if self.predictions[i] == self.correct_values[i]:
                if self.predictions[i] == 1:
                    self.true_pos += 1
                else:
                    self.true_neg += 1
            else:
                if self.predictions[i] == 1:
                    self.false_pos += 1
                else:
                    self.false_neg += 1
    
    def print_metrics(self):
        print("accuracy: ",self.accuracy())
        print("recall:   ",self.recall())
        print("precision:",self.precision())
        print("F1:       ",self.F1())
        
