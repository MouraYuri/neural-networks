import pandas as pd

def preProcessingData(data):
    for x in range(len(data)):
        data[x] = data[x].split(' ')
        data[x][2] = data[x][2].rstrip()
        data[x][0], data[x][1], data[x][2] = int(data[x][0]), int(data[x][1]), int(data[x][2])
    df = pd.DataFrame(data, columns = ['x1', 'x2', 'y'])
    return df

def trainWeights(dataset, threshold, eta):
    '''
    df.columns => return number of columns

    '''



dataset = preProcessingData(open('./simple-perceptron/dataset.txt', 'r').readlines())
print(dataset)

