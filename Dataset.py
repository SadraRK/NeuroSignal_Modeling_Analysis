import numpy as np
import glob
import torch
from torch_geometric.data import Data

def NormalizeData(Data, resolution):
    norm_Data = Data
    for i in range(0, Data.shape[2]):
        norm_Data[:, :, i] = np.around(Data[:, :, i] * 2, resolution-1)/2
    return  norm_Data

def load_data():
    A = np.zeros((82, 82, 999))
    B = np.zeros((82, 82, 999))
    L_gen = np.zeros((1, 1, 999))
    L_bs = np.zeros((1, 10, 999))

    mod1_k = 0
    for mod1_npy_file in glob.glob('modal1_networks/*.np[yz]'):
        temp = np.load(mod1_npy_file)
        if np.count_nonzero(temp) == 0:
            print(mod1_npy_file, mod1_k)
        else:
            A[:, :, mod1_k] = temp
            mod1_k = mod1_k + 1

    mod2_k = 0
    for mod2_npy_file in glob.glob('modal2_networks/*.np[yz]'):
        temp = np.load(mod2_npy_file)
        if np.count_nonzero(temp) == 0:
            print(mod2_npy_file, mod2_k)
        else:
            B[:, :, mod2_k] = abs(temp)
            mod2_k = mod2_k + 1

    bs_k = 0
    remove_idx = []
    for label_bs in glob.glob('labels/bs/*.np[yz]'):
        L_bs[0, :, bs_k] = np.load(label_bs)
        if len(np.where(np.isnan(L_bs[0, :, bs_k]))[0]) != 0:
            remove_idx.append(bs_k)
        bs_k = bs_k + 1

    gen_k = 0
    for label_gen in glob.glob('labels/gender/*.np[yz]'):
        temp = np.load(label_gen)
        L_gen[0, 0, gen_k] = temp
        gen_k = gen_k + 1
    return A, B, L_gen, L_bs, remove_idx

def Create_Dataset(in_Data, in_Label):
    Out_Dataset = []
    for i in range(in_Label.shape[2]):
        Out_Label = in_Label[:, :, i]
        Out_Dataset.append(Data(x= torch.ones((82, 1), dtype= torch.float), edge_index= torch.tensor(np.array(in_Data[1][i]), dtype= torch.long), edge_attr= torch.tensor(np.array(in_Data[0][i]), dtype= torch.float), y= torch.tensor(np.array(Out_Label), dtype= torch.float)))
    return Out_Dataset