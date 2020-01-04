import numpy as np
import pandas as pd

'''
Batch Gradient Descent
'''


class Adaline :
    def __init__(self, data, eta, n_iter):
        self.data = self.preProcessingData(data)
        self.weights = np.zeros(5)
        self.eta = eta
        self.n_iter = n_iter
        self.train()
    
    def preProcessingData(self, data):
        data = [x.split(',') for x in data]
        data = [[float(data[x][y]) for y in range(len(data[x]))] for x in range(len(data))]
        data = pd.DataFrame(data, columns = ['x1','x2','x3','x4', 'y'])
        return data

    def f(self, x):
        if x >=0: return 1
        else: return -1

    def sse(self):
        

    def train(self):
        for _ in range(10):
        #for _ in range(self.n_iter):
            sse, ctrl = 0, 0 #(SSE == Sum of Squared Errors)
            for Xj in range(len(self.data.columns) - 1):
                for i in range(len(self.data)):
                    row = np.append(-1, (self.data.iloc[i][:-1]).to_numpy())
                    #row = np.append(-1, row)
                    #sse += (self.data.iloc[x][-1])
                    ctrl += (self.data.iloc[i][-1] - np.inner(row, self.weights))*self.data.iloc[i][Xj]
                    print(ctrl)



data = open('./simple-perceptron/dataset2', 'r')
datatotest = open('./simple-perceptron/datatotest', 'r')
a = Adaline(data, 0.01, 50)