
import warnings
warnings.filterwarnings("ignore")

from sklearn.inspection import permutation_importance
import argparse
import numpy as np
from imblearn.combine import SMOTEENN 
import pandas as pd
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, mean_squared_error,make_scorer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def encode_preprocessor(X_train):
    numeric_cols = list(set(X_train) - set(X_train.select_dtypes('O')))
    categorical_cols = list(set(X_train.select_dtypes('O')))

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())  
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

def get_months(x):
    ym = x.split('and')
    if len(ym) == 2:
        years = int(ym[0].split()[0])
        months = int(ym[1].split()[0])
    elif len(ym) == 1 and 'month' in ym[0]:
        months = int(ym[0].split()[0])
        years = 0
    elif len(ym) == 1 and 'year' in ym[0]:
        years = int(ym[0].split()[0])
        months = 0
    return years*12 + months

def preprocess(df):
    df.drop(columns=['Unnamed: 0','policy_id'],axis=1,inplace=True)
    df['age_of_car']  = df['age_of_car'].apply(get_months)
    for col in df:
        df[col] = df[col].replace({'Yes':1,'No':0})
    return df

def main(train_file, test_file):
    my_id = 123
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    policy_ids = df_test["policy_id"]
    
    df_train['make'] = df_train['make'].astype('str')
    df_test['make'] = df_test['make'].astype('str')

    df_train = preprocess(df_train)
    df_test = preprocess(df_test)
    
    
    df_train['area'] = df_train['length']*df_train['width']*df_train['height']
    df_train.drop(columns=['length','width','height'],inplace=True)

    df_test['area'] = df_test['length']*df_test['width']*df_test['height']
    df_test.drop(columns=['length','width','height'],inplace=True)

    df_train[['max_torque_Nm', 'max_torque_rpm']] = df_train['max_torque'].str.split('@', expand=True)
    df_train['max_torque_Nm'] = df_train['max_torque_Nm'].str.extract(r'(\d+\.?\d*)').astype(float)
    df_train['max_torque_rpm'] = df_train['max_torque_rpm'].str.extract(r'(\d+)').astype(int)
    df_train.drop(columns=['max_torque'], inplace=True)

    df_test[['max_torque_Nm', 'max_torque_rpm']] = df_test['max_torque'].str.split('@', expand=True)
    df_test['max_torque_Nm'] = df_test['max_torque_Nm'].str.extract(r'(\d+\.?\d*)').astype(float)
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].str.extract(r'(\d+)').astype(int)
    df_test.drop(columns=['max_torque'], inplace=True)

    df_train[['max_power_bhp', 'max_power_rpm']] = df_train['max_power'].str.split('@', expand=True)
    df_train['max_power_bhp'] = df_train['max_power_bhp'].str.extract(r'(\d+\.?\d*)').astype(float)
    df_train['max_power_rpm'] = df_train['max_power_rpm'].str.extract(r'(\d+)').astype(int)
    df_train.drop(columns=['max_power'], inplace=True)

    df_test[['max_power_bhp', 'max_power_rpm']] = df_test['max_power'].str.split('@', expand=True)
    df_test['max_power_bhp'] = df_test['max_power_bhp'].str.extract(r'(\d+\.?\d*)').astype(float)
    df_test['max_power_rpm'] = df_test['max_power_rpm'].str.extract(r'(\d+)').astype(int)
    df_test.drop(columns=['max_power'], inplace=True)
    
    num_cols = list(set(df_train) - set(df_train.select_dtypes('O')))
    
    corr_matrix = df_train[num_cols].corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > 0.97)] 

    df_train.drop(to_drop, axis=1, inplace=True)
    df_test.drop(to_drop, axis=1, inplace=True)

    X_train = df_train.drop('age_of_policyholder',axis=1)
    y_train = df_train['age_of_policyholder']
    
    encoder = encode_preprocessor(X_train)
    X_train = encoder.fit_transform(X_train)
    
    gbr_param = {'n_estimators': 250, 'min_samples_split': 36, 
                 'min_samples_leaf': 39, 'max_features': 'sqrt',
                 'max_depth': 3, 'learning_rate': 0.1}
    
    reg = GradientBoostingRegressor(random_state=42,**gbr_param)
    reg.fit(X_train,y_train)
    X_test = df_test.drop('age_of_policyholder',axis=1)
    y_test = df_test['age_of_policyholder']

    X_test = encoder.transform(X_test)
    y_pred = np.round(reg.predict(X_test),1)
    mse = mean_squared_error(y_test,y_pred)
    print('MSE',mse)
    
    submission_df = pd.DataFrame({'policy_id': policy_ids, 'age': y_pred})
    submission_df.to_csv(f'z{my_id}.PART1.output.csv', index=False)

    #################  Classification ###############

#     df_train = df_train[cols]
#     df_test = df_test[cols]

    X_train = df_train.drop('is_claim',axis=1)
    y_train = df_train['is_claim']

    X_test = df_test.drop('is_claim',axis=1)
    y_test = df_test['is_claim']
    
    encoder = encode_preprocessor(X_train)
    
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)
    
    sampler = SMOTEENN(random_state=42)
    X_train, y_train = sampler.fit_resample(X_train,y_train)
    
    xg_param = {'n_estimators': 180, 'max_depth': 22, 'learning_rate': 0.05}
    
    clf = XGBClassifier(random_state=40,**xg_param,n_jobs=-1)
    clf.fit(X_train,y_train,eval_metric='map')
    pred = clf.predict(X_test)
    score = f1_score(y_test, pred,average='macro')

    print('F1_Score',score)
    submission_df = pd.DataFrame({'policy_id': policy_ids, 'is_claim': pred})
    submission_df.to_csv(f'z{my_id}.PART2.output.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process train and test CSV files.')
    parser.add_argument('train_file', type=str, help='Path to train CSV file')
    parser.add_argument('test_file', type=str, help='Path to test CSV file')
    args = parser.parse_args()
    main(args.train_file, args.test_file)
