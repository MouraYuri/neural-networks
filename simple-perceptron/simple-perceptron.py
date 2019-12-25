import pandas as pd
from random import random
import numpy as np

'''
    O que o perceptron tem que ter:
        pesos
        dataset com os exemplos




'''

class Percetron:
    
    def __init__(self, data, n):
        self.dataFrame = self.preProcessingData(data)
        self.weights = np.random.rand(1, 4)
        print(self.weights)
    
    def preProcessingData(self, data):
        data = [x.split(',') for x in data]
        data = pd.DataFrame(data, columns = ['x1','x2','x3','x4', 'y'])
        return data


data = open('./simple-perceptron/dataset2', 'r')
Percetron(data, 4)
