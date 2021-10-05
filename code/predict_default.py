import pickle

import numpy as np
import pandas as pd

DATASET_PATH = '../data/dataset.csv'
TARGET = "default"
REQUIRED_COLUMNS = ['account_amount_added_12_24m',
                    'age',
                    'merchant_category',
                    'merchant_group',
                    'has_paid',
                    'max_paid_inv_0_12m',
                    'name_in_email',
                    'num_active_inv',
                    'num_arch_dc_0_12m',
                    'num_arch_dc_12_24m',
                    'num_arch_ok_0_12m',
                    'num_arch_rem_0_12m',
                    'num_unpaid_bills',
                    'status_last_archived_0_24m',
                    'status_2nd_last_archived_0_24m',
                    'status_3rd_last_archived_0_24m',
                    'status_max_archived_0_6_months',
                    'status_max_archived_0_24_months',
                    'recovery_debt',
                    'sum_capital_paid_account_0_12m',
                    'sum_capital_paid_account_12_24m',
                    'sum_paid_inv_0_12m',
                    'time_hours']


def load_data(path):
    """Load data """
    print('Loading the data')
    df = pd.read_csv(path, delimiter=';')
    print('Successfully uploaded the data')
    return df


def prepare_data(df):
    """function to prepare data in order to pass as an input to our model"""

    print('Preparing the data')
    # 1. Check for default column in test data
    if TARGET in df.columns:
        df = df[df["default"].isna()].drop(["default"], axis=1).reset_index(drop=True)

    # 2. Check for all required columns
    for col_name in REQUIRED_COLUMNS:
        if col_name not in df.columns:
            print('Writing the data to csv file where required column values are missing')
            df[df[REQUIRED_COLUMNS].isnull().any(axis=1)].to_csv(
                '../artifacts/required_columns_values_missing.csv')
            raise Exception('Required column  is missing:{}', format(col_name))

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=REQUIRED_COLUMNS, inplace=True)

    # 3. Drop columns which had more than 50% data missing training data
    df.drop(["account_incoming_debt_vs_paid_0_24m",
                  "account_status",
                  "account_worst_status_0_3m",
                  "account_worst_status_12_24m",
                  "account_worst_status_3_6m",
                  "account_worst_status_6_12m",
                  "avg_payment_span_0_3m",
                  "worst_status_active_inv"], axis=1, inplace=True)

    # 3. Impute data for columns which had less than 30% missing numerical columns:
    dataset = load_data(DATASET_PATH)
    col_list = ["account_days_in_dc_12_24m",
                "account_days_in_rem_12_24m",
                "account_days_in_term_12_24m",
                "avg_payment_span_0_12m",
                "num_active_div_by_paid_inv_0_12m",
                "num_arch_written_off_0_12m",
                "num_arch_written_off_12_24m"]

    for col in col_list:
        col_vals = df[col]
        dataset_col_vals = dataset[col]
        if sum(col_vals.isnull()) != 0:
            df[col] = col_vals.fillna(dataset_col_vals.median())

    # 3. Drop correlated columns'
    for col_name in df.columns:
        if col_name in ['max_paid_inv_0_24m', 'num_arch_ok_12_24m',
                        'status_max_archived_0_12_months']:
            df.drop([col_name], axis=1, inplace=True)

    # 4.Categorical column data preparation
    df_categorical = df[["merchant_category", "merchant_group", "name_in_email"]]
    # load one hot encoder from disk
    one_hot_encoder = pickle.load(open('../artifacts/one_hot_encoder.pkl', 'rb'))
    df_one_hot = one_hot_encoder.transform(df_categorical)
    df_one_hot = pd.DataFrame(df_one_hot.toarray())
    df.reset_index()
    df_one_hot.reset_index()
    df_final = pd.concat([df.drop(["merchant_category", "merchant_group", "name_in_email"], axis=1).reset_index(),
                          df_one_hot], axis=1).drop(['index'], axis=1)

    print('Data preparation finished successfully')

    return df_final


def predict(df_final):
    """This function is used to predict the default and write the result to prediction.csv file"""

    # load model from disk
    default_predictor_rf = pickle.load(open('../artifacts/default_predictor_rf.pkl', 'rb'))

    if 'uuid' in df_final.columns:
        df_final.to_csv("test.csv")

        df_final['default_prediction'] = default_predictor_rf.predict(
            df_final.drop(["uuid"], axis=1))
        prediction = df_final[['uuid', 'default_prediction']]


    else:
        df_final['default_prediction'] = default_predictor_rf.predict(df_final)
        prediction = df_final['default_prediction']

    prediction.to_csv('../artifacts/prediction.csv')

    print('Successfully predicted the data, please check: ../artifacts/prediction.csv')
    return prediction


if __name__ == "__main__":
    # 1. Load Data
    df = load_data(DATASET_PATH)
    # 2. Data prepartion
    df_final = prepare_data(df)
    # 3. Prediction
    predict(df_final)
