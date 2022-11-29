import numpy as np
import random

def adj2c00(batches):
    edge_weight = []
    edge_index = []
    batch_index = np.zeros((batches.shape[2]))
    for i in range(batches.shape[2]):
        temp_data = batches[:, :, i]
        edge_index.append(np.nonzero(temp_data))
        edge_weight.append(np.reshape(temp_data[edge_index[i]], (len(temp_data[edge_index[i]]), 1)))
        batch_index[i] = i
    return [edge_weight, edge_index, batch_index]

def Split_Data(in_Data, splitting_Ratio):
    random.shuffle(in_Data)
    train_num = round(splitting_Ratio*len(in_Data))
    test_num = len(in_Data) - train_num

    TrainData = in_Data[0:train_num]
    TestData = in_Data[train_num:]
    return TrainData, TestData