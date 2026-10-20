# 加载数据
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
import shap
import os
from pathlib import Path

class Model_1(nn.Module): #spectral branch
    def __init__(self, input_dim, output_dim, hidden_layer_sizes_1):
        """
        params:
        input_dim          : The dimension of the input data
        output_dim         : The dimension of the output data
        hidden_layer_sizes : The number and dimensions of the hidden layers in LSTM
        """
        super().__init__()
        self.num_layers = len(hidden_layer_sizes_1)  
        self.model_layers = nn.ModuleList()  
        self.model_layers.append(nn.Conv1d(input_dim, hidden_layer_sizes_1[0], kernel_size=3, padding=1))
        self.model_layers.append(nn.ReLU())
        for i in range(1, self.num_layers):
            self.model_layers.append(nn.Conv1d(hidden_layer_sizes_1[i-1], hidden_layer_sizes_1[i], kernel_size=3, padding=1))
            self.model_layers.append(nn.ReLU())
            
    def forward(self, input_seq):
        input_seq=input_seq[:,:,1:] 
        mod_out = input_seq.permute(0,2,1) 
        for mod in self.model_layers:
            mod_out = mod(mod_out)  
        predict = mod_out
        return predict

class Model_2(nn.Module): #temporal branch
    def __init__(self, input_dim, output_dim, hidden_layer_sizes_2):
        """
        params:
        input_dim          : The dimension of the input data
        output_dim         : The dimension of the output data
        hidden_layer_sizes : The number and dimensions of the hidden layers in LSTM
        """
        super().__init__()
        self.num_layers = len(hidden_layer_sizes_2)  
        self.model_layers = nn.ModuleList()  
        self.model_layers.append(nn.Conv1d(input_dim, hidden_layer_sizes_2[0], kernel_size=3, padding=1))
        self.model_layers.append(nn.ReLU())
        for i in range(1, self.num_layers):
            self.model_layers.append(nn.Conv1d(hidden_layer_sizes_2[i-1], hidden_layer_sizes_2[i], kernel_size=3, padding=1))
            self.model_layers.append(nn.ReLU())

    def forward(self, input_seq):
        mod_out = input_seq 
        mod_out = mod_out[:,:,np.array([0]+np.arange(1,509,10).tolist())] 
        for mod in self.model_layers:
            mod_out = mod(mod_out)  
        predict = mod_out
        return predict

# spectro-temporal ParallelNet
class ParallelNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels, hidden_layer_sizes_1, hidden_layer_sizes_2):
        super().__init__()

        self.feature_branch = Model_1(
            input_dim=input_dim, 
            output_dim=output_dim,
            hidden_layer_sizes_1=hidden_layer_sizes_1,
        )

        self.temporal_branch = Model_2(
            input_dim=num_channels, 
            output_dim=output_dim,
            hidden_layer_sizes_2=hidden_layer_sizes_2,
        )

        fusion_dim = (int(input_dim/10)+2+num_channels)*32  
        self.norm = nn.LayerNorm(fusion_dim) 
        self.head = nn.Linear(fusion_dim, output_dim)

    def forward(self, x):
        # x: [B, W, F] 
        h_feat = self.feature_branch(x)     # [B, W, F]
        h_time = self.temporal_branch(x)    # [B, W, F]
        fused = torch.cat([h_feat, h_time], dim=-1)
        last_hidden = fused.reshape(fused.shape[0], -1)
        out = self.head(last_hidden)     # [B, out_dim]
        return out

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

#(1) Define the structure and parameters of the GRU model
#(1.1) Define model parameters
input_dim = 508   # The input dimension is 508 dimensions.
hidden_layer_sizes_1 = [256, 128, 64, 32]  # LSTM layer structure
hidden_layer_sizes_2 = [16, 32]  # LSTM layer structure
output_dim = 1   # Output dimension is 1
#(1.2) Define the loss function and the optimization function
loss_function = nn.MSELoss() # loss
learn_rate = 0.0002
epochs = 200

#(2)Load train_loader,val_loader, and test_loader
batch_size,workers = 32,0 # "Workers" represents the number of parallel processes.
rootdir1=r'../../step8_DeepSHAP/CNN_CNN_timestamp'
rootdir2=r'../DeepSHAP'
name=['fold_1','fold_2','fold_3']
test_set,test_label=load(os.path.join(rootdir1,'test_set')),load(os.path.join(rootdir1,'test_label'))
scaler_1=load(os.path.join(rootdir1,'scaler_1'))
# Every time, specific indicators must be selected!!!!!!!!!!!!!!!!!!!
sor_idx=0
test_label=test_label[:,sor_idx].reshape(-1,1)
# Set an array to store the R2 and RMSE values of the train, val, and test sets for the three-fold split; set pred_test_list
sta_npy=np.zeros((3,7)) #3:3 fold;7:rmse_c,r2_c,rmse_v,r2_v,rmse_t,r2_t,rpd_t;
shap_values_list=[] # Store the test set results for each discount.
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
    
    #(3) Model training set, validation set for validation, test set for testing
    #(3.1) Training
    model = ParallelNet(input_dim, output_dim, train_set.shape[1], hidden_layer_sizes_1, hidden_layer_sizes_2)  
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)  # 优化器
    model_train(model, epochs, loss_function, optimizer, train_loader, val_loader, device, i)
    #(3.2) Load the optimal model for the current fold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('best_model_'+str(i)+'.pt',weights_only=False)
    model = model.to(device)
    #(3.3) Using DeepSHAP to explain neural networks
    explainer = shap.DeepExplainer(model, torch.cat([train_set,val_set],dim=0))
    shap_values = explainer.shap_values(test_set,check_additivity=False)
    shap_values = np.squeeze(np.abs(shap_values))
    shap_values_list.append(shap_values)
#Take the average of the shap_values_list for the three folds.
mean_shap_values=np.mean(shap_values_list,axis=0)
np.save(os.path.join(rootdir2,'mean_shap_values.npy'),mean_shap_values) 