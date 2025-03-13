import numpy as np
import pandas as pd
import torch
import pickle
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

LAG = 12 # We are predicting in 5-min intervals for the next 1 hour (60 min)
HORIZON_WINDOW = 24 # We are using 5-min windows over 2 hours (120 min) for training

class SmoothedTimeseriesDataset(Dataset): # Subclassing from Dataset
                                          # in order to do preprocessing
    def __init__(self,data,targets, length, sigma):
        self.data = data
        self.targets = targets
        self.length = length # horizon window
        self.sigma = sigma

    def __len__(self):
        
        return len(self.data) - (self.length)
    
    def step6_preprocessing(self, x):
        # Apply Gaussian filter on timeseries data
        
        # your code goes here
        
        return x
    
    def __getitem__(self, idx):
        x, y = self.data[idx:idx+self.length].copy(), self.targets[idx+self.length].copy()
        x = self.step6_preprocessing(x)
        
        return x, y

class GlucoseData: # Preparing data for learning 
                   # using either present or historical data
    def __init__(self, path_to_data= "week2pkls/present_day_data.pkl"):
        with open(path_to_data, "rb") as file:
            self.present_day_data = pickle.load(file)
    
    def step2_preprocessing(self, df):
        
        # your code goes here
        new_df = df.resample('1s')
        return new_df
    
    def step3_4_preprocessing(self, df):
        
        # your code goes here
        new_df = df.ffill()
        new_df.fillna(0)
        return new_df
    
    def step5_preprocessing(self, df):
        
        # your code goes here
#         df.loc[len(df)] = pd.Series(dtype='float64')
#         df.date
        return df

    def lag_target(self, df, lag=LAG, target_col='Glucose'):
        target_df = pd.concat([df[target_col].shift(periods=-i) for i in range(1, lag + 1)], axis=1).dropna(axis=0)
        df = df.iloc[:len(target_df)]

        return df, target_df
            
    def make_loader(self, X, Y, horizon_window=HORIZON_WINDOW, sampling_rate=1, batch_size=1):
        dataset = SmoothedTimeseriesDataset(data=X, targets=Y.shift(1).values, length=horizon_window,sigma=1)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        return dataloader
    
    def generate_training_data(self,
                               path_to_glucose_X_data='week2pkls/glucose_X_2a.csv',
                               path_to_glucose_y_data='week2pkls/glucose_y_2a.csv'):
        
        # Data preprocessing steps:
        # 1. Save all the blood glucose timestamps
        # 2. Resample the features to a time delta of 1 second
        # 3. Forward fill the missing values by using the last available values
        # 4. Fill the left missing values with 0
        # 5. Resample the features to a time delta of 5 minutes
        # 6. Smooth each feature with a 1D Gaussian filter over a window containing the past 2 hours of data
        
        X_df = pd.DataFrame()
        y_df = pd.DataFrame()
        
        X_lists = []
        y_lists = []

        for patient_id in self.present_day_data.keys():
            df_train = self.present_day_data[patient_id].copy().rename(columns={'glucose_level':'Glucose'})
            df_train['Glucose'] = df_train[['Glucose']].interpolate('linear')
            df_train = self.step2_preprocessing(df_train)
            df_train = self.step3_4_preprocessing(df_train)
            df_train = self.step5_preprocessing(df_train)
            df_train = df_train.astype(np.float32)
            df_train.dropna(inplace=True)
            X_train, Y_train = self.lag_target(df_train[['Glucose']])
            train_loader = self.make_loader(X_train.values, Y_train)

            for data in train_loader:

                initial = np.concatenate((np.expand_dims(np.array([patient_id]), axis=0), data[0][0].T), axis=1)
                X_lists.append(np.squeeze(initial))
                y_lists.append((data[1][0].tolist()))

        columns = ['patient_id'] + [f't{i}' for i in range(0, 120, 5)]
        X = pd.DataFrame(X_lists, columns=columns)

        columns = [f't{i}' for i in range(120, 180, 5)]
        y = pd.DataFrame(y_lists, columns=columns)
        
        X_df = pd.concat([X_df, X])
        y_df = pd.concat([y_df, y])
        
        X_df.to_csv(path_to_glucose_X_data, index=False)
        y_df.to_csv(path_to_glucose_y_data, index=False)
        
        return X_df, y_df

class TrainSplits:
    def __init__(self,data_X,data_y,combine=True,
            pid_to_indices = {}):
        
        # Create training, validation & test dataframes for all patients

        self.train_df_X = pd.DataFrame()
        self.validate_df_X = pd.DataFrame()
        self.test_df_X = pd.DataFrame()
        self.train_df_y = pd.DataFrame()
        self.validate_df_y = pd.DataFrame()
        self.test_df_y = pd.DataFrame()

        self.data_dicts = {}
        self.split_data_by_person(data_X,data_y,
                    pid_to_indices)
        if combine:
            self.combine(pid_to_indices)

    def split_data(self,data_X,data_y):
        stop_train = int(.8 * data_X.shape[0])
        stop_validate = int(.9 * data_X.shape[0])
        X_train =data_X[:stop_train]
        X_validate = data_X[stop_train:stop_validate]
        X_test = data_X[stop_validate:]

        y_train =data_y[:stop_train]
        y_validate = data_y[stop_train:stop_validate]
        y_test = data_y[stop_validate:]
        return X_train,X_validate,X_test,\
               y_train,y_validate,y_test

    def split_data_by_person(self,data_X,data_y,
            pid_to_indices):

        for pid,indices in pid_to_indices.items():
            data_splits = self.split_data(
                    data_X.loc[indices],
                    data_y.loc[indices]
                    )
            self.data_dicts[pid]=data_splits
            
    def combine(self,pid_to_indices):
        
        for pid in pid_to_indices:
            data_splits = self.data_dicts[pid]
            self.train_df_X = pd.concat([self.train_df_X, data_splits[0]])
            self.validate_df_X = pd.concat([self.validate_df_X, data_splits[1]])
            self.test_df_X = pd.concat([self.test_df_X, data_splits[2]])
            self.train_df_y = pd.concat([self.train_df_y, data_splits[3]])
            self.validate_df_y = pd.concat([self.validate_df_y, data_splits[4]])
            self.test_df_y = pd.concat([self.test_df_y, data_splits[5]])
            
        self.train_df_X.drop(columns=['patient_id'], inplace=True)
        self.validate_df_X.drop(columns=['patient_id'], inplace=True)
        self.test_df_X.drop(columns=['patient_id'], inplace=True)

class Diabetes():
    def __init__(self, path_to_X_data ='week2pkls/glucose_X_2a.csv', 
                       path_to_y_data ='week2pkls/glucose_y_2a.csv'):
        self.X = pd.read_csv(path_to_X_data).astype(np.float32)
        self.y = pd.read_csv(path_to_y_data).astype(np.float32)
