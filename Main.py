import numpy as np
from Dataset import load_data, NormalizeData, Create_Dataset
from DataSplit import adj2c00, Split_Data
from Models import RUN_GCN

if __name__ == '__main__':
    A, B, L_gen, L_bs, remove_idx = load_data()
    A = NormalizeData(A, 5)
    B = NormalizeData(B, 5)
    L_bs = NormalizeData(L_bs, 5)

    Modal1_Data = adj2c00(A)
    Modal2_Data = adj2c00(B)

    Modal1_Data_Classification_Dataset = Create_Dataset(Modal1_Data, L_gen)
    Modal1_Data_Regression_Dataset = Create_Dataset(adj2c00(np.delete(A, remove_idx, 2)), np.delete(L_bs, remove_idx, 2))
    Modal1_Data_Classification_Train_Data,  Modal1_Data_Classification_Test_Data = Split_Data(Modal1_Data_Classification_Dataset, 0.75)
    Modal1_Data_Regression_Train_Data, Modal1_Data_Regression_Test_Data = Split_Data(Modal1_Data_Regression_Dataset, 0.75)

    Modal2_Data_Classification_Dataset = Create_Dataset(Modal2_Data, L_gen)
    Modal2_Data_Regression_Dataset = Create_Dataset(adj2c00(np.delete(B, remove_idx, 2)), np.delete(L_bs, remove_idx, 2))
    Modal2_Data_Classification_Train_Data,  Modal2_Data_Classification_Test_Data = Split_Data(Modal2_Data_Classification_Dataset, 0.75)
    Modal2_Data_Regression_Train_Data, Modal2_Data_Regression_Test_Data = Split_Data(Modal2_Data_Regression_Dataset, 0.75)

    # ////////////////////////////////////////////////////////////////////////// GCN MODEL ////////////////////////////////////////////////////////////////////////// #
    # acc_modal1_classification = RUN_GCN(0, Modal1_Data_Classification_Train_Data, Modal1_Data_Classification_Test_Data, 0)
    # acc_modal1_regression = RUN_GCN(0, Modal1_Data_Regression_Train_Data, Modal1_Data_Regression_Test_Data, 0)
    # acc_modal2_classification = RUN_GCN(0, Modal2_Data_Classification_Train_Data, Modal2_Data_Classification_Test_Data, 0)
    # acc_modal2_regression = RUN_GCN(0, Modal2_Data_Regression_Train_Data, Modal2_Data_Regression_Test_Data, 0)
    # print(f'GCN modal1 Classification Accuracy: {max(acc_modal1_classification): 04f}', f'GCN modal1 Regression Accuracy: {max(acc_modal1_regression): 04f}')
    # print(f'GCN modal2 Classification Accuracy: {max(acc_modal2_classification): 04f}', f'GCN modal2 Regression Accuracy: {max(acc_modal2_regression): 04f}')

    # ////////////////////////////////////////////////////////////////////////// SAGPooling MODEL ////////////////////////////////////////////////////////////////////////// #
    # acc_modal1_classification = RUN_GCN(1, Modal1_Data_Classification_Train_Data, Modal1_Data_Classification_Test_Data, 0)
    # acc_modal1_regression = RUN_GCN(1, Modal1_Data_Regression_Train_Data, Modal1_Data_Regression_Test_Data, 0)
    # acc_modal2_classification = RUN_GCN(1, Modal2_Data_Classification_Train_Data, Modal2_Data_Classification_Test_Data, 0)
    # acc_modal2_regression = RUN_GCN(1, Modal2_Data_Regression_Train_Data, Modal2_Data_Regression_Test_Data, 0)
    # print(f'SAGCN modal1 Classification Accuracy: {max(acc_modal1_classification): 04f}', f'GCN modal1 Regression Accuracy: {max(acc_modal1_regression): 04f}')
    # print(f'SAGCN modal2 Classification Accuracy: {max(acc_modal2_classification): 04f}', f'GCN modal2 Regression Accuracy: {max(acc_modal2_regression): 04f}')

    # ////////////////////////////////////////////////////////////////////////// CustomGCN MODEL ////////////////////////////////////////////////////////////////////////// #
    # acc_modal1_classification = RUN_GCN(2, Modal1_Data_Classification_Train_Data, Modal1_Data_Classification_Test_Data, 0)
    # acc_modal1_regression = RUN_GCN(2, Modal1_Data_Regression_Train_Data, Modal1_Data_Regression_Test_Data, 0)
    # acc_modal2_classification = RUN_GCN(2, Modal2_Data_Classification_Train_Data, Modal2_Data_Classification_Test_Data, 0)
    # acc_modal2_regression = RUN_GCN(2, Modal2_Data_Regression_Train_Data, Modal2_Data_Regression_Test_Data, 0)
    # print(f'CustomGCN modal1 Classification Accuracy: {max(acc_modal1_classification): 04f}', f'GCN modal1 Regression Accuracy: {max(acc_modal1_regression): 04f}')
    # print(f'CustomGCN modal2 Classification Accuracy: {max(acc_modal2_classification): 04f}', f'GCN modal2 Regression Accuracy: {max(acc_modal2_regression): 04f}')