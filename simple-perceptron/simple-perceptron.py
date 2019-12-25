import pandas as pd
from random import random
import numpy as np

'''
    O que o perceptron tem que ter:
        pesos
        dataset com os exemplos




'''

class Percetron:
    
    def __init__(self, data, n, eta, eras):
        self.dataFrame = self.preProcessingData(data)
        self.weights = np.random.rand(1, 5)
        self.eta = eta
        self.eras = eras
        self.train()

    def f(self, x):
        if x >=0: return 1
        else: return 0
    
    def preProcessingData(self, data):
        data = [x.split(',') for x in data]
        data = [[float(data[x][y]) for y in range(len(data[x]))] for x in range(len(data))]
        data = pd.DataFrame(data, columns = ['x1','x2','x3','x4', 'y'])
        return data

    def train(self):
        for _ in range(self.eras):
            totalError = 0
            for x in range(len(self.dataFrame)):
                row = (self.dataFrame.iloc[x, :-1]).to_numpy()
                row = np.append(row, 1)
                yHat = self.f(np.inner(row, self.weights))
                e = int(self.dataFrame.iloc[x, -1:]) - yHat
                if e != 0:
                    totalError +=1
                    for y in range(len(self.weights)):
                        self.weights[y] = self.weights[y] + self.eta*e*(row[y])
            self.dataFrame = self.dataFrame.sample(frac=1)

    def test(self, values):
        return self.f(np.inner(self.weights, values))

data = open('./simple-perceptron/dataset2', 'r')
p = Percetron(data, 4, 0.1, 30)
print(p.test([5.7, 3.8, 1.7, 0.3, 1]))
print(p.test([7.9, 3.8, 6.4, 2.0, 1]))
