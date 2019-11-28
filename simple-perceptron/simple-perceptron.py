import pandas as pd
from random import random
import numpy as np
def preProcessingData(data):
    for x in range(len(data)):
        data[x] = data[x].split(' ')
        data[x][2] = data[x][2].rstrip()
        data[x][0], data[x][1], data[x][2] = int(data[x][0]), int(data[x][1]), int(data[x][2])
    df = pd.DataFrame(data, columns = ['x1', 'x2', 'y'])
    return df

def train(dataset, threshold, eta, eras):
    #assigning random values to weights and assigning value 1 to theta
    weightsVector, theta = np.array([]), 1
    for x in range(len(dataset.columns)-1):
        weightsVector = np.append(weightsVector, [random()])
    
    #training start
    for era in range(eras):
        for i in range(len(dataset)):
            yHat = (np.inner(dataset.iloc[i, :len(dataset.columns)-1], weightsVector) + theta) #estimated y | Provável bucho aqui ein
            E = dataset.iloc[i, len(dataset.columns)-1]-yHat
            
            #Updating weights and theta
            for x in range(len(weightsVector)):
                weightsVector[x] = weightsVector[x] -  eta*(2*E*(-(dataset.iloc[i, x])))
            theta = theta - eta*(-2*E)
        
        #Shuffling the dataset
        dataset.sample(frac=1)
    print(weightsVector)
dataset = preProcessingData(open('./simple-perceptron/dataset.txt', 'r').readlines())
train(dataset, 0.5, 0.1, 10)

