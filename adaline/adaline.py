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


    def sse(self): #Sum of Squared Errors
        ctrl = 0
        for i in range(len(self.data)):
            row, y = np.append(-1, (self.data.iloc[i][:-1]).to_numpy()), self.data.iloc[i][-1]
            linearOutput = np.inner(row, self.weights)
            ctrl += ((y - linearOutput)**2)/2
        return ctrl
        

        

    def train(self):
        for _ in range(self.n_iter):
            print('========================================')
            self.data = self.data.sample(frac=1) #shuffling the dataset
            weights = self.weights
            
            #Updating weights
            for w in range(len(self.weights)):
                deltaW = 0
                for i in range(len(self.data)):
                    row = np.append(-1, self.data.iloc[i][:-1])
                    linearOutput = np.inner(row, weights) 
                    y = self.data.iloc[i][-1]
                    deltaW += (y - linearOutput)*row[w]
                self.weights[w] += self.eta*deltaW
                print('sse => ', self.sse())
        print('weights final => ',self.weights)
        print('sse final => ', self.sse())

    def test(self, data):
        return np.inner(self.weights, data)



data = open('./adaline/datasetAdaline', 'r')
datatotest = open('./adaline/datasetAdalineTest', 'r')
a = Adaline(data, 0.001, 100)
print(a.test([-1,5.8,2.7,4.1,1.0]))