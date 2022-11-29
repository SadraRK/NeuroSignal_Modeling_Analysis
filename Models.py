import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import cross_entropy, mse_loss
from torch.nn import Linear
from torch_geometric.utils import softmax
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, SAGPooling
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
from torch.utils.data import Subset

class model_GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_inputs, output_channels):
        super(model_GCN, self).__init__()
        torch.manual_seed(0)
        self.conv1 = GCNConv(input_channels, hidden_inputs)
        self.conv2 = GCNConv(hidden_inputs, hidden_inputs)
        self.conv3 = GCNConv(hidden_inputs, hidden_inputs)
        self.out = Linear(hidden_inputs, output_channels)

    def forward(self, x, edge_idx, edge_weight, batch, active):
        x = self.conv1(x= x, edge_index = edge_idx, edge_weight= edge_weight)
        x = active(x)
        x = self.conv2(x, edge_idx, edge_weight)
        x = active(x)
        x = self.conv3(x, edge_idx, edge_weight)
        x = active(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.out(x)
        return F.log_softmax(x, dim=1)

class model_SagPoolGCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_inputs, output_channels):
        super(model_SagPoolGCN, self).__init__()
        torch.manual_seed(0)
        self.sagpool = SAGPooling(in_channels=input_channels, GNN=GCNConv)
        self.conv1 = GCNConv(1, hidden_inputs)
        self.conv2 = GCNConv(hidden_inputs, hidden_inputs)
        self.conv3 = GCNConv(hidden_inputs, hidden_inputs)
        self.out = Linear(hidden_inputs, output_channels)

    def forward(self, x, edge_idx, edge_weight, batch, active):
        y = self.sagpool(x, edge_idx, edge_weight, batch)
        x = active(y[0])
        x = self.conv1(x= x, edge_index = y[1], edge_weight= y[2])
        x = active(x)
        x = self.conv2(x= x, edge_index = y[1], edge_weight= y[2])
        x = active(x)
        x = self.conv3(x= x, edge_index = y[1], edge_weight= y[2])
        x = active(x)
        x = global_mean_pool(x, y[3])
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.out(x)
        return F.log_softmax(x, dim=1)

class model_CustomGCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_inputs, output_channels):
        super(model_CustomGCN, self).__init__()
        torch.manual_seed(0)
        self.l1 = Linear(input_channels, hidden_inputs)
        self.l2 = Linear(hidden_inputs, hidden_inputs)
        self.l3 = Linear(hidden_inputs, hidden_inputs)
        self.out = Linear(hidden_inputs, output_channels)

    def forward(self, x, batch, active):
        x = self.l1(x)
        x = active(x)
        x = self.l2(x)
        x = active(x)
        x = self.l3(x)
        x = active(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.out(x)
        return F.log_softmax(x, dim=1)

def start_train(device, model, loss_function, active, optimizer, trainData):
    model.train(True)
    loss_all = 0
    for data in trainData:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(x= data.x, edge_idx= data.edge_index, edge_weight= data.edge_attr, batch= data.batch, active= active)
        loss = loss_function(out, data.y.to(device))
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    model.train(False)
    return loss_all / len(trainData.dataset)

def test_model(model, device, loss_function, active, testData):
    model.eval()
    correct = []
    for data in testData:
        data = data.to(device)
        out = model(x= data.x, edge_idx= data.edge_index, edge_weight= data.edge_attr, batch= data.batch, active= active)
        pred = out.detach().cpu().numpy()
        label = data.y.detach().cpu().numpy()
        if (data.y.shape[1] == 1):
            cout = int(sum((pred == label)))
            correct.append(cout)
        else:
            correct.append(loss_function(out, data.y.to(device)).detach().cpu().numpy())
    if (data.y.shape[1] == 1):
        cout = int(sum(correct)) / len(testData.dataset)
    else:
        cout = np.mean(correct)
    return cout

def custom_train(device, model, model_sub, loss_function, active, optimizer, trainData):
    model.train(True)
    loss_all = 0
    for data in trainData:
        data = data.to(device)
        optimizer.zero_grad()
        out1 = model_sub(x= data.x, edge_idx= data.edge_index, edge_weight= data.edge_attr, batch= data.batch, active= active)
        out2 = model_sub(x= data.x, edge_idx= data.edge_index, edge_weight= data.edge_attr, batch= data.batch, active= active)
        x = torch.column_stack((out1, out2))
        out = model(x= x, batch= torch.tensor(np.linspace(0, len(out1) - 1, len(out1)), dtype= int), active= active)

        loss = loss_function(out, data.y.to(device))
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    model.train(False)
    return loss_all / len(trainData.dataset)

def custom_test(model, model_sub, device, loss_function, active, testData):
    model.eval()
    correct = []
    for data in testData:
        data = data.to(device)
        out1 = model_sub(x= data.x, edge_idx= data.edge_index, edge_weight= data.edge_attr, batch= data.batch, active= active)
        out2 = model_sub(x= data.x, edge_idx= data.edge_index, edge_weight= data.edge_attr, batch= data.batch, active= active)
        x = torch.column_stack((out1, out2))
        out = model(x= x, batch= torch.tensor(np.linspace(0, len(out1) - 1, len(out1)), dtype= int), active= active)
        pred = out.detach().cpu().numpy()
        label = data.y.detach().cpu().numpy()
        if (data.y.shape[1] == 1):
            cout = int(sum((pred == label)))
            correct.append(cout)
        else:
            correct.append(loss_function(out, data.y.to(device)))
    if (data.y.shape[1] == 1):
        cout = int(sum(correct)) / len(testData.dataset)
    else:
        cout = correct[-1]
    return cout

def RUN_GCN(model_type, Training_Data, Test_Data, ACT_Func):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == 0:
        model = model_GCN(input_channels= 1, hidden_inputs= 64, output_channels= Training_Data[0].y.shape[1]).to(device)
    elif model_type==1:
        model = model_SagPoolGCN(input_channels=1, hidden_inputs=64, output_channels=Training_Data[0].y.shape[1]).to(device)
    elif model_type == 2:
        model = model_CustomGCN(input_channels=2, hidden_inputs=64, output_channels=Training_Data[0].y.shape[1]).to(device)
        model_sub = model_SagPoolGCN(input_channels=1, hidden_inputs=64, output_channels=Training_Data[0].y.shape[1]).to(device)
    Activation_Function_List = [F.relu, F.sigmoid, F.tanh, F.silu, F.gelu]
    actFunc = Activation_Function_List[ACT_Func]
    Loss_Function = [torch.nn.BCELoss(), torch.nn.MSELoss()]
    if (Training_Data[0].y.shape[1] == 1):
        lossFunc = Loss_Function[0]
    else:
        lossFunc = Loss_Function[1]
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.01)

    Training_idx = np.linspace(0, len(Training_Data), len(Training_Data) + 1).astype(int)
    Test_idx = np.linspace(0, len(Test_Data), len(Test_Data) + 1).astype(int)
    Training_Data = Subset(Training_Data, Training_idx)
    Test_Data = Subset(Test_Data, Test_idx)
    train_D = DataLoader(Training_Data.dataset, batch_size= 64, shuffle= False)
    test_D = DataLoader(Test_Data.dataset, batch_size= 64, shuffle=False)

    train_acc = []
    test_acc = []
    temp_model = []
    print(lossFunc, actFunc)
    print('Start Training...')
    if model_type != 2:
        for epoch in range(0, 150):
            loss = start_train(device, model, lossFunc, actFunc, optimizer, train_D)
            temp_model.append(model)
            train_acc.append(test_model(model, device, lossFunc, actFunc, train_D))
            test_acc.append(test_model(model, device, lossFunc, actFunc, test_D))
            print(f'Training Epoch #{epoch:03d}', f'Training Loss: {loss:05f}, ', f'Training Accuracy: {train_acc[epoch]:05f}, ', f'Test Accuracy: {test_acc[epoch]:05f}')

    else:
        for epoch in range(0, 150):
            loss = custom_train(device, model, model_sub, lossFunc, actFunc, optimizer, train_D)
            temp_model.append(model)
            train_acc.append(custom_test(model, model_sub, device, lossFunc, actFunc, train_D))
            test_acc.append(custom_test(model, model_sub, device, lossFunc, actFunc, test_D))
            print(f'Training Epoch #{epoch:03d}', f'Training Loss: {loss:05f}, ', f'Training Accuracy: {train_acc[epoch]:05f}, ', f'Test Accuracy: {test_acc[epoch]:05f}')
    return test_acc, train_acc