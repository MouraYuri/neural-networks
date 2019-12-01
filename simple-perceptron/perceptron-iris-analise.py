import pandas as pd

def preProcessingData(dataset):
    return pd.DataFrame(dataset, columns=['x1', 'x2', 'x3', 'x4', 'y'])

def train(dataset):
    


dataset = preProcessingData([x.split(',') for x in (open('./simple-perceptron/irisDataset.txt', 'r')).readlines()])

