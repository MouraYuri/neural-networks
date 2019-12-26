import pandas as pd
from random import random
import numpy as np

'''
    O que o perceptron tem que ter:
        pesos
        dataset com os exemplos




'''

class Percetron:
    
    def __init__(self, data, n, eta):
        self.dataFrame = self.preProcessingData(data)
        self.weights = np.random.rand(1, 5)
        self.eta = eta
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
        for _ in range(100):
        #while(True):
            totalError=0
            for x in range(len(self.dataFrame)):
                row = (self.dataFrame.iloc[x, :-1]).to_numpy()
                row = np.append(row, 1)
                yHat = self.f(np.inner(row, self.weights))
                e = int(self.dataFrame.iloc[x, -1:]) - yHat
                if e != 0: totalError += 1
                for y in range(len(self.weights)):
                    self.weights[y] += self.eta*e*(row[y])
            hits = len(self.dataFrame) - totalError
            hitsPercentage = hits/len(self.dataFrame)
            print('hits => {}\nhitsPercentage => {}'.format(hits, hitsPercentage))
            print(self.weights)
            if totalError == 0: break
            self.dataFrame = self.dataFrame.sample(frac=1)

    def test(self, values):
        return self.f(np.inner(self.weights, values))

    def test_dataset(self, data):
        dfToTest, totalError = (self.preProcessingData(data)).sample(frac=1), 0
        for x in range(len(dfToTest)):
            row = (dfToTest.iloc[x, :-1]).to_numpy()
            row = np.append(row, 1)
            yHat = self.test(row)
            e = int(dfToTest.iloc[x, -1:]) - yHat
            if (e!=0): totalError+=1
        hits = len(dfToTest) - totalError
        hitsPercentage = hits/len(dfToTest)
        print('hits => {}\nhitsPercentage => {}'.format(hits, hitsPercentage))

data = open('./simple-perceptron/dataset2', 'r')
datatotest = open('./simple-perceptron/datatotest', 'r')
p = Percetron(data, 4, 0.01)
print('===================================================')
p.test_dataset(datatotest)