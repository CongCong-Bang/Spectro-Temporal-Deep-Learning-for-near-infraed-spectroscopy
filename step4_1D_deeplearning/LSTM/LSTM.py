import torch
from joblib import dump, load
import torch.utils.data as Data
import numpy as np
import pandas as pd
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font", family='Microsoft YaHei')
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from thop import profile,clever_format

#(*1) Define the model structure
class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels, hidden_layer_sizes):
        """
        params:
        input_dim          : The dimension of the input data
        output_dim         : The dimension of the output data
        hidden_layer_sizes : The number and dimensions of the hidden layers in LSTM
        """
        super().__init__()

        self.output_dim = output_dim
        self.num_channels = num_channels
        self.num_layers = len(hidden_layer_sizes)  
        self.lstm_layers = nn.ModuleList()  
        self.lstm_layers.append(nn.LSTM(input_dim, hidden_layer_sizes[0], batch_first=True))
        self.lstm_layers.append(nn.BatchNorm1d(1)) 
        for i in range(1, self.num_layers):
                self.lstm_layers.append(nn.LSTM(hidden_layer_sizes[i-1], hidden_layer_sizes[i], batch_first=True))
                self.lstm_layers.append(nn.BatchNorm1d(1))
        self.linear  = nn.Linear(hidden_layer_sizes[-1]*self.num_channels, output_dim)

    def forward(self, input_seq):
        #Change the input shape, LSTM adapts to the network input [batch, seq_length, H_in]
        lstm_out = input_seq
        for i in np.arange(len(self.lstm_layers)):
            if i%2==0:
                lstm_out, _= self.lstm_layers[i](lstm_out)   
            elif i%2==1:
                lstm_out= self.lstm_layers[i](lstm_out)  
        predict = lstm_out.reshape(lstm_out.shape[0], -1)      
        predict = self.linear(predict)
        
        return predict
    
###########################################################################
#(*2) Define the model training process
def model_train(model, epochs, loss_function, optimizer, train_loader, val_loader, device, idx):
    model = model.to(device)
    # the minimum mse
    minimum_mse = 1000.
    # optimal model
    best_model = model
    train_mse = []     # The average value of the MSE index changes for each epoch recorded on the training set
    val_mse = []      # Record the average change in the MSE metric for each epoch on the validation set

   # Computational model running time
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_mse_loss = []    # Save the MSE loss of the current epoch
        for seq, labels in train_loader: 
            seq, labels = seq.to(device), labels.to(device)
            # Zero the gradients and initialize before each parameter update
            optimizer.zero_grad()
            # Forward propagation
            y_pred = model(seq)  #   torch.Size([16, 10])
            # Loss calculation
            loss = loss_function(y_pred, labels)
            train_mse_loss.append(loss.item()) # Calculate the MSE loss
            # Backpropagation and parameter update
            loss.backward()
            optimizer.step()
        # Calculate the total loss
        train_av_mseloss = np.average(train_mse_loss) 
        train_mse.append(train_av_mseloss)
        
        print(f'Epoch: {epoch+1:2} train_MSE-Loss: {train_av_mseloss:10.8f}')
        # After each epoch, the experimental results are verified on the validation set
        with torch.no_grad():
            # Set the model to evaluation mode
            model.eval()
            val_mse_loss = []    # Save the MSE loss of the current epoch
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                pre = model(data)
                # Loss calculation
                val_loss = loss_function(pre, label)
                val_mse_loss.append(val_loss.item()) 
            
            # Calculate the total loss
            val_av_mseloss = np.average(val_mse_loss) 
            val_mse.append(val_av_mseloss)
            print(f'Epoch: {epoch+1:2} val_MSE_Loss:{val_av_mseloss:10.8f}')
            # If the current model's MSE is lower than the previous best accuracy rate, then update the best model.
            # Save the current optimal model parameters
            if val_av_mseloss < minimum_mse:
                minimum_mse = val_av_mseloss
                best_model = model # Update the parameters of the best model
    
    # Save the best parameters
    torch.save(best_model, 'best_model_'+str(idx)+'.pt')
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')
    
    # Visualization
    plt.plot(range(epochs), train_mse, color = 'b',label = 'train_MSE-loss')
    plt.plot(range(epochs), val_mse, color = 'y',label = 'test_MSE-loss')
    plt.legend()
    plt.show()   
    print(f'min_MSE: {minimum_mse}')

#(1) Define the structure and parameters of the LSTM model
#(1.1) Define model parameters
input_dim = 508   # The input dimension is 508 dimensions.
hidden_layer_sizes = [256, 128, 64, 32]  # LSTM layer structure
output_dim = 1   # Output dimension is 1
#(1.2) Define the loss function and the optimization funtion
loss_function = nn.MSELoss() # loss
learn_rate = 0.0002
epochs = 200

#(2) Load train_loader,val_loader, and test_loader
batch_size,workers = 32,0 # "Workers" represents the number of parallel processes.
rootdir1=r'../data_preparation'

name=['fold_1','fold_2','fold_3']
test_set,test_label=load(os.path.join(rootdir1,'test_set')),load(os.path.join(rootdir1,'test_label'))
scaler_1=load(os.path.join(rootdir1,'scaler_1'))
scaler_2=load(os.path.join(rootdir1,'scaler_2'))

# Every time, specific indicators must be selected!!!!!!!!!!!!!!!!!!!
sor_idx=0
test_label=test_label[:,sor_idx].reshape(-1,1)
temp_scaler_2=scaler_2
# Set an array to store the R2 and RMSE values of the train, val, and test sets for the three-fold split; set pred_test_list
sta_npy=np.zeros((3,7)) #3:3 fold;7:rmse_c,r2_c,rmse_v,r2_v,rmse_t,r2_t,rpd_t;
pre_test_list=[] # Store the test set results for each discount.
for i in range(len(name)): # Carry out training and validation for each fold separately
    print(str(i)+'fold training')
    # The test datasets are all the same, but the training set and validation set for each fold are different.
    train_set=load(os.path.join(rootdir1,name[i],'train_set'))
    train_label=load(os.path.join(rootdir1,name[i],'train_label'))
    val_set=load(os.path.join(rootdir1,name[i],'val_set'))
    val_label=load(os.path.join(rootdir1,name[i],'val_label'))
    train_label,val_label=train_label[:,sor_idx].reshape(-1,1),val_label[:,sor_idx].reshape(-1,1)
    
    # Package the data in the form of a loader
    train_loader = Data.DataLoader(dataset=Data.TensorDataset(train_set, train_label),
                                   batch_size=batch_size, num_workers=workers, drop_last=False)
    val_loader = Data.DataLoader(dataset=Data.TensorDataset(val_set, val_label),
                                   batch_size=batch_size, num_workers=workers, drop_last=False)
    test_loader = Data.DataLoader(dataset=Data.TensorDataset(test_set, test_label),
                                  batch_size=batch_size, num_workers=workers, drop_last=False)
    torch.manual_seed(100)  # Set a random seed to ensure the reproducibility of the experimental results.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    #(3)Model training set, validation set for validation, test set for testing
    #(3.1) Training
    model = LSTMModel(input_dim, output_dim, train_set.shape[1], hidden_layer_sizes)  #Instantiate the LSTM model; train_set.shape[1] represents the length of the time window
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)
    model_train(model, epochs, loss_function, optimizer, train_loader, val_loader, device, i)
    #(3.2) Load the optimal model for the current fold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('best_model_'+str(i)+'.pt',weights_only=False)
    model = model.to(device)
    #(3.3) Training set
    ori_data_1,pre_data_1=[],[]
    with torch.no_grad():
        for data, label in train_loader:
            model.eval()  # Set the model to evaluation mode
            origin_label = label.tolist()
            ori_data_1 += origin_label
            # Prediction
            data, label = data.to(device), label.to(device)
            train_pred = model(data)  
            # Use .squeeze() to convert it into a one-dimensional tensor
            train_pred = train_pred.tolist()
            pre_data_1 += train_pred
    ori_data_1 = np.array(ori_data_1)  
    pre_data_1 = np.array(pre_data_1)  
    train_score = r2_score(ori_data_1, pre_data_1) # The R2 coefficient of the training set
    ori_data_1=temp_scaler_2.inverse_transform(ori_data_1) # Scaler inverse standardization (true value)
    pre_data_1=temp_scaler_2.inverse_transform(pre_data_1) # Scaler inverse standardization (predicted value)
    train_mse = mean_squared_error(ori_data_1, pre_data_1) 
    train_rmse = np.sqrt(train_mse) # Coefficient of RMSE for the training set
    print('R2 in training set--R2: ',train_score)
    print('RMSE in training set--RMSE: ',train_rmse)
    #(3.4) Validation set
    ori_data_2,pre_data_2=[],[]
    with torch.no_grad():
        for data, label in val_loader:
            model.eval()  # Set the model to evaluation mode
            origin_label = label.tolist()
            ori_data_2 += origin_label
            # Prediction
            data, label = data.to(device), label.to(device)
            val_pred = model(data)  
            # Use .squeeze() to convert it into a one-dimensional tensor
            val_pred = val_pred.tolist()
            pre_data_2 += val_pred
    ori_data_2 = np.array(ori_data_2)  
    pre_data_2 = np.array(pre_data_2)  
    val_score = r2_score(ori_data_2, pre_data_2) # The R2 coefficient of the validation set
    ori_data_2=temp_scaler_2.inverse_transform(ori_data_2) # Scaler inverse standardization (true value)
    pre_data_2=temp_scaler_2.inverse_transform(pre_data_2) # Scaler inverse standardization (predicted value)
    val_mse = mean_squared_error(ori_data_2, pre_data_2) 
    val_rmse = np.sqrt(val_mse) # Coefficient of RMSE for the validation set
    print('R2 in validation set--R2: ',val_score)
    print('RMSE in validation set--RMSE: ',val_rmse)
    #(3.5) Testing set
    ori_data_3,pre_data_3=[],[]
    with torch.no_grad():
        for data, label in test_loader:
            model.eval()  # Set the model to evaluation mode
            origin_label = label.tolist()
            ori_data_3 += origin_label
            # Prediction
            data, label = data.to(device), label.to(device)
            test_pred = model(data)
            # Use .squeeze() to convert it into a one-dimensional tensor
            test_pred = test_pred.tolist()
            pre_data_3 += test_pred
    ori_data_3 = np.array(ori_data_3)  
    pre_data_3 = np.array(pre_data_3)  
    test_score = r2_score(ori_data_3, pre_data_3) #The R2 coefficient of the testing set
    ori_data_3=temp_scaler_2.inverse_transform(ori_data_3) #Scaler inverse standardization (true value)
    pre_data_3=temp_scaler_2.inverse_transform(pre_data_3) #Scaler inverse standardization (predicted value)
    test_mse = mean_squared_error(ori_data_3, pre_data_3) 
    test_rmse = np.sqrt(test_mse) #测试集的RMSE系数
    test_rpd = np.std(ori_data_3)/test_rmse
    print('R2 in testing set--R2: ',test_score)
    print('RMSE in testing set--RMSE: ',test_rmse)
    
    #(3.6)Store the statistical results in sta_npy and pre_test_list
    sta_npy[i,0]=train_score
    sta_npy[i,1]=train_rmse
    sta_npy[i,2]=val_score
    sta_npy[i,3]=val_rmse
    sta_npy[i,4]=test_score
    sta_npy[i,5]=test_rmse
    sta_npy[i,6]=test_rpd
    pre_test_list.append(pre_data_3) #Store the test set results in the variable pre_test_list

#(4)Present the statistical results, the curve of predicted values versus true values in the test set
#(4.1)Statistical results
aver_sta=np.mean(sta_npy,axis=0)
print(aver_sta)
#(4.2)The curve of predicted values versus true values of the test set
pre_test_list=np.mean(np.array(pre_test_list),axis=0)
# for testing batch b1
plt.figure(figsize=(12, 6), dpi=100)
plt.plot(ori_data_3[:34], label='Original',color='orange')  # True values
plt.plot(pre_test_list[:34], label='LSTM Predictions',color='green')  # Predicted values
plt.legend()
plt.savefig('Testing batch b1_'+str(sor_idx)+'.png')
plt.show()
# for testing batch b2
plt.figure(figsize=(12, 6), dpi=100)
plt.plot(ori_data_3[34:], label='Original',color='orange')  # True values
plt.plot(pre_test_list[34:], label='LSTM Predictions',color='green')  # Predicted values
plt.legend()
plt.savefig('Testing batch b2_'+str(sor_idx)+'.png')
plt.show()