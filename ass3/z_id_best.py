from sklearn.metrics import r2_score
import numpy as np
import sys
import pandas as pd
import warnings
from sklearn import linear_model
import copy
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.metrics import average_precision_score
from sklearn import preprocessing
from sklearn.metrics import f1_score

# ignore warning
# warnings.simplefilter('ignore', np.RankWarning)
np.seterr(divide='ignore', invalid='ignore')

PART1_OUTPUT = "z_id.PART1.output.csv"
PART2_OUTPUT = "z_id.PART2.output.csv"


# how well the relationship between the values
def check_relationship_score(d1, d2):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')

        degs = [5, 4, 3, 2, 1]
        for deg in degs:
            try:
                relationship_model = np.poly1d(np.polyfit(d1, d2, deg))
                return r2_score(d2, relationship_model(d1))
            except np.RankWarning:
                pass

    print("Warning: np.RankWarning")
    return 0

# define a function to apply to the Yes/No column
def yes_no_function(value):
    if value == "Yes":
        return 1
    else:
        return 0

# define a function to apply to the age_of_car column
def age_of_car_function(age):
    res = [int(i) for i in age.split() if i.isdigit()]
    if "years" in age and "months" in age:
        return res[0] * 12 + res[1]
    elif "years" in age:
        return res[0] * 12
    elif "months" in age:
        return res[0]
    else:
        return 0


def preprocess(df):
    del df[df.columns[0]]  # remove the first column

    # convert string to number
    for series_name, series in df.items():

        if series_name == 'policy_id':
            pass
        elif series_name == 'is_claim':
            pass
        elif series_name == 'age_of_car':
            df['age_of_car'] = df['age_of_car'].apply(age_of_car_function)
        elif series_name.startswith('is_'):
            # print(series_name)
            df[series_name] = df[series_name].apply(yes_no_function)
        else:
            # print(series_name, "Data type: ", series.dtype)
            if "int" in series.dtype.name or "float" in series.dtype.name:
                pass
            else:
                le = preprocessing.LabelEncoder()
                le.fit(df[series_name])
                df[series_name] = le.transform(df[series_name])

    return df


# choose columns as training
def choose_cols(df, checked_col, score):
    chosen_cols = []  # chosen columns to train

    for col_name in df:
        if col_name == 'policy_id':
            pass
        else:
            try:
                d1 = df[col_name]
                d2 = df[checked_col]

                x = check_relationship_score(d1, d2)
                #print(x)
                if x >= score:
                    chosen_cols.append(col_name)
                    # print(series_name, x)
            except:
                pass

    return chosen_cols


# part 1: regression
def regression(train_df_in, test_df_in):

    # preprocess
    train_df = preprocess(train_df_in)
    test_df = preprocess(test_df_in)

    # keep policy id for output
    policy_ids = test_df["policy_id"].values

    # true (used to check Mean Square Error)
    age_of_policyholder_true = test_df["age_of_policyholder"].values

    sample_cols = choose_cols(train_df, "age_of_policyholder", 0.0002)
    sample_cols.remove('age_of_policyholder')

    print('Info: chosen cols = ', len(sample_cols), sample_cols)

    # ignore the age_of_policyholder in test dataframe

    test_cols_df = test_df[sample_cols]
    liner_regression = linear_model.LinearRegression()
    liner_regression.fit(train_df[sample_cols].values, train_df['age_of_policyholder'].values)

    result = pd.DataFrame(columns=['policy_id', 'age'])

    for index, row in test_cols_df.iterrows():
        x = row.tolist()
        predicted_age_of_policyholder = liner_regression.predict([x])
        result.loc[index] = [policy_ids[index], predicted_age_of_policyholder[0]]
        # print(predicted_age_of_policyholder)

    result.to_csv(PART1_OUTPUT, encoding='utf-8', index=False)

    # guess
    age_of_policyholder_pred = result["age"].values

    # print mean_squared_error
    print("Info: Mean Square Error:", mean_squared_error(age_of_policyholder_true, age_of_policyholder_pred))

# part 2
def classification(train_df_in, test_df_in):

    # preprocess
    train_df = preprocess(train_df_in)
    test_df = preprocess(test_df_in)

    is_claim_train = train_df['is_claim'].values

    # keep policy id for output
    policy_ids = test_df["policy_id"].values

    # true (used to check Macro Average Precision)
    is_claim_true = test_df["is_claim"].values

    sample_cols = ['policy_tenure', 'age_of_car', 'age_of_policyholder', 'area_cluster', 'population_density', 'make', 'segment', 'model', 'ncap_rating', 'is_parking_sensors']

    from sklearn.preprocessing import StandardScaler
    scale = StandardScaler()

    train_df = copy.deepcopy(train_df[sample_cols])
    test_df = copy.deepcopy(test_df[sample_cols])
    train_df['e'] = train_df.sum(axis=1, numeric_only=True)
    test_df['e'] = test_df.sum(axis=1, numeric_only=True)

    print('Info: chosen cols = ', len(sample_cols), sample_cols)

    x = train_df[sample_cols].values
    scaledX = scale.fit_transform(x)

    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    classifier = classifier.fit(scaledX, is_claim_train)

    result = pd.DataFrame(columns=['policy_id', 'is_claim'])

    test_cols_df = test_df[sample_cols]
    for index, row in test_cols_df.iterrows():
        x = row.tolist()
        scaled = scale.transform([x])

        predicted_claim = classifier.predict([scaled[0]])
        result.loc[index] = [policy_ids[index], predicted_claim[0]]

    result.to_csv(PART2_OUTPUT, encoding='utf-8', index=False)

    # guess
    is_claim_pred = result["is_claim"].values

    # print Macro Average Precision
    print("Info: f1_score:", f1_score(is_claim_true, is_claim_pred, average='macro'))

# usage eg.: python3 current_py_file train.csv test.csv
if __name__ == '__main__':
    # check valid arguments
    if len(sys.argv) != 3:
        print("Missing train and test files")
        exit(0)
    origin_train_df = pd.read_csv(sys.argv[1])
    origin_test_df = pd.read_csv(sys.argv[2])

    # part 1
    print("Part 1")
    regression(copy.deepcopy(origin_train_df), copy.deepcopy(origin_test_df))

    # part 2
    print("\nPart 2")
    classification(origin_train_df, origin_test_df)
