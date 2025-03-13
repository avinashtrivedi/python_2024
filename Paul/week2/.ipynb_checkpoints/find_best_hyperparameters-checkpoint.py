import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.autograd import Variable
import random

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

NUM_EPOCHS = 50
LEARNING_RATE = 0.001
NUM_LAYERS = 1 # Number of LSTM layers
NUM_CLASSES = 12 # Number of output classes [t+5, t+10, t+15...t+60]
HORIZON_WINDOW = 24
TIMES = {
    '5': 0,
    '10': 1,
    '15': 2,
    '20': 3,
    '25': 4,
    '30': 5,
    '35': 6,
    '40': 7,
    '45': 8,
    '50': 9,
    '55': 10,
    '60': 11
    }
MIN_DELTA = 0.0001
MIN_LR = 1e-5

class LSTM_model(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers):
        super(LSTM_model, self).__init__()
        self.num_classes = num_classes #number of classes
        self.hidden_size = hidden_size # number of features in hidden state
        self.num_layers = num_layers #number of lstm layers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv1d(1, 8, kernel_size=24, padding='same') # in_channels, out_channels, kernel_size
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(8, 16,kernel_size=12, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2)        
        self.conv3 = nn.Conv1d(16, 32,kernel_size=6, padding='same')
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=False) #lstm
        
        self.fc_1 =  nn.Linear(hidden_size,256) #fully connected 1
        self.fc_2 =  nn.Linear(256,32) #fully connected 2
        self.fc_3 =  nn.Linear(32,12) #fully connected 3

        torch.nn.init.xavier_uniform_(self.fc_1.weight)
        torch.nn.init.xavier_uniform_(self.fc_2.weight)
        torch.nn.init.xavier_uniform_(self.fc_3.weight)
        
    def forward(self,x):
        # print(x.shape) # N, 1, 24
        
        out  = self.conv1(x) # default activation is None; out.shape: N, 8, 24
        out = self.pool1(out) # out.shape: N, 8, 12
        
        out = self.conv2(out) # out.shape: N, 16, 12
        out = self.pool2(out) # out.shape: N, 16, 6
        
        out = self.conv3(out) # out.shape: N, 32, 6
        out = self.pool3(out) # out.shape: N, 32, 3
        
        h_0 = Variable(torch.zeros(self.num_layers, out.size(0), self.hidden_size)).to(self.device)
        c_0 = Variable(torch.zeros(self.num_layers, out.size(0), self.hidden_size)).to(self.device)

        output, (hn, cn) = self.lstm(out, (h_0, c_0)) # hn.shape = 1, N, 64
        hn = hn.view(-1, self.hidden_size) # hn.shape = N, 64

        out = self.fc_1(hn) # Dense, out.shape = N, 256
        out = self.fc_2(out) # out.shape = N, 32
        out = self.fc_3(out) # out.shape = N, 12
        return out

class PatientDataset(Dataset):
    def __init__(self,data,targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        x, y = self.data[idx].copy(), self.targets[idx].copy()
        return x, y


def make_loader(X, Y, batch_size=1024):
    
    X_val = np.expand_dims(X.values, -1)
    dataset = PatientDataset(data=X_val, targets=Y.values)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(self, optimizer, patience=3, min_lr=MIN_LR, factor=0.1):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        # take one step of the learning rate scheduler while providing the validation loss as the argument
        self.lr_scheduler.step(val_loss)

            

def best_lstm_parameters_training_data(X_train, y_train, X_validate, y_validate):
    """ Find the best hyperparameters for a model. 
    
    Parameters:
    X_train (Numpy Array-like): training data.
    y_train (Numpy Array-like): training labels.
    X_validate (Numpy Array-like): Validation data.
    y_validate (Numpy Array-like): Validation labels.
    
    Returns:
    best_model : Best performing model
    best_hidden_size : Best hyperparameter
    min_val_loss : Best validation loss
    """
    print('X_train',type(X_train))
    
    # this should return the best model by searching for the best hidden layer size,
    # the best hidden layer size and the associated validation loss
    # return None, None, None
    
    set_seed() # do not change seed for autograder purposes
    
    best_model = None
    best_hidden_size = None
    min_val_loss = np.inf
    
    
    # TODO: your code goes here. Fill up the missing pieces. 
    for i in range(2, 8):
        hidden_size = 2**i
        print('Hidden layer size: ' + str(hidden_size))

        lstm_model =  LSTM_model(num_classes=NUM_CLASSES, hidden_size=hidden_size, num_layers=NUM_LAYERS) # TODO: Instantiate LSTM_model class 
        lstm_model = lstm_model.to(device)
        
        criterion =  nn.L1Loss() # TODO: Define L1 Loss
        optimizer =  torch.optim.RMSprop(lstm_model.parameters(), lr=LEARNING_RATE)# TODO: Define RMSProp optimizer  

        train_loader = make_loader(X_train, y_train)
        val_loader = make_loader(X_validate, y_validate)

        val_losses = []
        train_losses = []
        
        # TODO: Write a training loop that iterate over NUM_EPOCHS 
        for epoch in range(NUM_EPOCHS):
            for i, data in enumerate(train_loader):
                lstm_model.train()  
                inputs, labels = data
                inputs = inputs.permute([0,2,1]).to(device)
                optimizer.zero_grad()
                outputs = lstm_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        

        # Evaluating the model performance (NOTE that we do it outside of the NUM_EPOCHS loop)
        print("Evaluating the performance ...")
        lstm_model.eval()
        for inputs, targets in val_loader:
            inputs = inputs.permute([0,2,1]).to(device)
            outputs = lstm_model.forward(inputs)
            targets = targets.to(device)
            val_loss = criterion(outputs, targets)
            val_losses.append(val_loss.item())

        with torch.no_grad():        
            val_epoch_loss = np.mean(val_losses)
            print("Hidden Size: {}".format(hidden_size),
                  "Val Loss: {:.6f}".format(val_epoch_loss))

            if val_epoch_loss < min_val_loss:
                min_val_loss = val_epoch_loss
                best_hidden_size = hidden_size
                best_model = lstm_model
                

    print('\nbest hyperparameters: ' + 'hidden layer size= ' + str(best_hidden_size))    
    
    return best_model, best_hidden_size, min_val_loss


def best_lstm_parameters_training_data_return_loss(X_train, y_train, X_validate, y_validate, hidden_size):
    """ 
    Returns training loss, validation loss and model trained on hidden size 64 
    
    Parameters:
    X_train (Numpy Array-like): training data.
    y_train (Numpy Array-like): training labels.
    X_validate (Numpy Array-like): Validation data.
    y_validate (Numpy Array-like): Validation labels.
    hidden_size (INT): Hidden size
    
    Returns:
    train_loss : A Python list of the average training loss for each epoch
    val_loss : A Python list of the average validation loss for each epoch
    lstm_model : Trained model
    """
    
    # returns training losses, validation losses, and model trained on best hyperparameters
    # remove the next line after you have instantiated LSTM model below
    return None, None, None

    set_seed() # do not change seed for autograder purposes
    
    val_losses = []
    train_losses = []
    
    print('Hidden layer size: ' + str(hidden_size))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    lstm_model =  None    # TODO: Instantiate LSTM_model class 
    lstm_model = lstm_model.to(device)

    criterion =  None     # TODO: Define L1 Loss
    optimizer =  None     # TODO: Define RMSProp optimizer  

    train_loader = None   # TODO: Define Dataset loader for training data 
    val_loader = None     # TODO: Define Dataset loader for validation data

    for epoch in range(NUM_EPOCHS):
        lstm_model.train()        
        # TODO: Training Loop

        
        
        lstm_model.eval()
        # TODO: Validation Loop
        
        

    return train_losses,val_losses,lstm_model



def best_lstm_parameters_training_data_with_lr(X_train, y_train, X_validate, y_validate,hidden_size):
    """
    Returns training, validation loss and model trained on hidden size 64, and
    uses learning rate scheduler to reduce overfitting.
    
    Parameters:
    X_train (Numpy Array-like): training data.
    y_train (Numpy Array-like): training labels.
    X_validate (Numpy Array-like): Validation data.
    y_validate (Numpy Array-like): Validation labels.
    hidden_size (INT): Hidden size
    
    Returns:
    train_loss : A Python list of the average training loss for each epoch
    val_loss : A Python list of the average validation loss for each epoch
    lstm_model : Trained model
    
    """
    
    # returns training losses, validation losses, and model trained on best hyperparameters
    # remove the next line after you have instantiated LSTM model below
    return None, None, None

    set_seed() # Do not change
        
    print('Hidden layer size: ' + str(hidden_size))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lstm_model =  None    # TODO: Instantiate LSTM_model class 
    lstm_model = lstm_model.to(device)

    criterion =  None     # TODO: Define L1 Loss
    optimizer =  None     # TODO: Define RMSProp optimizer  

    train_loader = None   # TODO: Define Dataset loader for training data 
    val_loader = None     # TODO: Define Dataset loader for validation data

    lr_scheduler = None   # TODO: Define learning rate scheduler. 


    for epoch in range(NUM_EPOCHS):
        lstm_model.train()
        epoch_val_losses = []
        epoch_train_losses = []
        # TODO: Training Loop

        
        
        lstm_model.eval()
        # TODO: Validation Loop, Note that to call lr_scheduler variable appropriately
        
        

    return train_losses,val_losses,lstm_model 

def compute_metric(df_pred):
    # Getting true labels and predicted for each [5, 10, 15, ... 60] minute
    # window, then calculate RMSE and MAE at each and put into a dataframe.
    rmse_by_time = []
    mae_by_time = []
    for time_idx, time in TIMES.items():
        pred_col = f"pred_cgm_{time}"

        mse = mean_squared_error(df_pred[pred_col], df_pred['glucose_level'])
        rmse_by_time.append(np.sqrt(mse))

        mae = mean_absolute_error(df_pred[pred_col], df_pred['glucose_level'])
        mae_by_time.append(mae)

    return pd.DataFrame({"Times": list(TIMES.keys()), "RMSE": rmse_by_time, "MAE": mae_by_time})

def predict_glucose_level_for_patient_i(best_model, X_test_i, y_test_i):
    # Use make_loader on test data to find predictions for the glucose level
    
    test_loader = make_loader(X_test_i, y_test_i)
    best_model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_pred = None
    x_test = []
    for inputs, _ in test_loader:

        inp_x_t0 = inputs[:,0:1,:].squeeze().data.numpy().tolist()
        x_test.extend(inp_x_t0)
        inputs = inputs.permute([0,2,1])
        inputs = inputs.to(device)
        
        y_pred = best_model.forward(inputs)
        y_pred = y_pred.cpu().data.numpy()
        if test_pred is None:
            test_pred = y_pred
        else:
            test_pred = np.concatenate((test_pred, y_pred))

    df_test = pd.DataFrame({"glucose_level": x_test})
    df_pred = df_test.copy()

    for time_idx, time in TIMES.items(): # time_idx is '5', '10'
        pred_col = f"pred_cgm_{time}"
        df_pred[pred_col] = np.nan
        df_pred[pred_col][HORIZON_WINDOW:] = test_pred[:-HORIZON_WINDOW, time]
        
    return df_pred

def rmse_from_training_population_testing_individual_best_lstm(best_model, data_dicts):
    """
    Calculate the RMSE for each patient in a testing population given a trained model.
    
    Parameters: 
    best model : A trained PyTorch model.
    data_dicts : A dictionary containing patient data, where keys are patient IDs and
                values are a tuple of training, validation and testing data.
    
    Returns:
    results_dict (dict) : A dictionary mapping patient IDs to tuple of the form
      (rmse for 30 min interval, list containing rmse for 5, 10, ..., 60 min intervals)
    
    
    Note that, some of the prediction from the best model can result in NaN values, 
    make sure to remove them using the **dropna()** method before proceeding to calculate the metric
    """
    
    results_dict = {}
    # your code goes here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    best_model.eval()
    for k in data_dicts.keys():
        df = data_dicts[k][2]
        df = df.drop(columns=['patient_id'])
        inputs = df.to(device)
        outputs = best_model.forward(inputs)
        results_dict[k] = outputs
        
    return results_dict



def best_lstm_parameters_train_individual_test_individual(data_dicts):
    """
    Train and evaluate the model on each patient data, return the predicted glucose level
    
    Parameters:
    data_dicts : A dictionary containing patient data, where keys are patient IDs and
                values are a tuple of training, validation and testing data.

    Returns:
    results_dict (dict) : A dictionary mapping patient IDs to tuple of the form
      (rmse for 30 min interval, list containing rmse for 5, 10, ..., 60 min intervals)
    
    Note that, you can use **best_lstm_parameters_training_data** function directly to avoid
    duplicating the work.
    """
    
    results_dict = {}
    
    
    # your code goes here
    
    
    return results_dict


def best_lstm_parameters_training_data_transfer(X_train_historical,
                                                y_train_historical, 
                                                X_val_historical, 
                                                y_val_historical,
                                                path_to_saved_model):

    """
    Train the model on the data provided and returns the best hidden
    size also make sure to save the best model. 
    
    Parameters:
    X_train_historical (Numpy Array-like): training data.
    y_train_historical (Numpy Array-like): training labels.
    X_validate_historical (Numpy Array-like): Validation data.
    y_validate_historical (Numpy Array-like): Validation labels.
    path_to_saved_model (str): Path to save the best model
    
    Returns:
    best_hidden_size : The best hidden size.
    
    Note:- We will be comparing the validation loss at the last epoch
    to get the best_model. 
    Directly use the best_lstm_parameters_training_data_with_lr function
    to avoid duplicating the work.
    """

    min_val_loss = np.inf
    best_hidden_size = None
    best_model = None
    
    for i in range(4, 8):
        hidden_size = 2**i
    
        #TODO: Your code goes here
    
    
    return None # remember to save the best model using torch.save()


def rmse_from_pre_training_historical_population_testing_individual_best_lstm(
                            best_hidden_size,
                            path_to_saved_model,
                            data_dicts_present_day):
    
    """
    Returns glucose prediction when the model is trained on historical data and tested on
    each patient data.
    
    Parameters:
    best_hidden_size (INT): Best hidden size.
    path_to_saved_model (str): Path of saved model trained on historical data.
    data_dicts_presents_data : A dictionary containing patient data, where keys are patient IDs and
                values are a tuple of training, validation and testing data.
                
    Returns:
    results_dict (dict) : A dictionary mapping patient IDs to tuple of the form
      (rmse for 30 min interval, list containing rmse for 5, 10, ..., 60 min intervals)
    
    
    """
    result_dict = {}
    # returns result dictionary, remove the next line after you have instantiated LSTM model below
    return None
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lstm_model = None       # TODO: Instantiate LSTM model
    lstm_model = lstm_model.to(device)
    criterion = None        # TODO: Define L1 Loss
    optimizer = None        # TODO: Define RMSProp Optimizer

    for pid in data_dicts_present_day.keys():

        # TODO: Define training, validation and testing data for each patient.

        
        model = lstm_model
        # TODO: Load the pretrained weight to the "model" 

        
        print("Transfer learning, patient ID:", pid)
        val_losses = []
        train_losses = []
        
        train_loader = None   # TODO: Define training loader 
        val_loader = None     # TODO: Define Validation loader
        
        lr_scheduler = None   # TODO: Instantiate Learing rate scheduler 
        
        for epoch in range(NUM_EPOCHS): 
            model.train()
            
            # TODO: Training Loop

            
            model.eval()
            # TODO: Validation Loop, Note that to call lr_scheduler variable appropriately

            
        
        # TODO: Predicting the glucose level for each patient using the model trained on test data
        # HINT: Some of the prediction from the model can results in NaN values, make sure to remove 
        # them using the **dropna()** method before proceeding to calculate the metric
    
    
    return result_dict

    
