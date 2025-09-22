from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"] = working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan})
    working_val_df["DAYS_EMPLOYED"] = working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan})
    working_test_df["DAYS_EMPLOYED"] = working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan})


    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.
    # Identify categorical columns of dtype object
    cat_cols = working_train_df.select_dtypes(include=["object"]).columns.tolist()

    # Find which categorical columns have 2 unique values and which have >2
    binary_cat_cols = []
    multi_cat_cols = []
    for col in cat_cols:
        n_unique = working_train_df[col].nunique(dropna=False)
        if n_unique == 2:
            binary_cat_cols.append(col)
        elif n_unique > 2:
            multi_cat_cols.append(col)

    # Ordinal encode binary categorical columns


    if binary_cat_cols:
        ordinal_encoder = OrdinalEncoder()
        working_train_df[binary_cat_cols] = ordinal_encoder.fit_transform(working_train_df[binary_cat_cols])
        working_val_df[binary_cat_cols] = ordinal_encoder.transform(working_val_df[binary_cat_cols])
        working_test_df[binary_cat_cols] = ordinal_encoder.transform(working_test_df[binary_cat_cols])

    # One-hot encode multi-category columns
    if multi_cat_cols:
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        # Fit on train
        onehot_encoder.fit(working_train_df[multi_cat_cols])

        # Transform all sets
        train_ohe = onehot_encoder.transform(working_train_df[multi_cat_cols])
        val_ohe = onehot_encoder.transform(working_val_df[multi_cat_cols])
        test_ohe = onehot_encoder.transform(working_test_df[multi_cat_cols])

        # Get new column names for one-hot features
        ohe_feature_names = onehot_encoder.get_feature_names_out(multi_cat_cols)

        # Convert to DataFrame for easy concat
        train_ohe_df = pd.DataFrame(train_ohe, columns=ohe_feature_names, index=working_train_df.index)
        val_ohe_df = pd.DataFrame(val_ohe, columns=ohe_feature_names, index=working_val_df.index)
        test_ohe_df = pd.DataFrame(test_ohe, columns=ohe_feature_names, index=working_test_df.index)

        # Drop original multi-category columns and concat new one-hot columns
        working_train_df = working_train_df.drop(columns=multi_cat_cols)
        working_val_df = working_val_df.drop(columns=multi_cat_cols)
        working_test_df = working_test_df.drop(columns=multi_cat_cols)

        working_train_df = pd.concat([working_train_df, train_ohe_df], axis=1)
        working_val_df = pd.concat([working_val_df, val_ohe_df], axis=1)
        working_test_df = pd.concat([working_test_df, test_ohe_df], axis=1)

    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.



    simple_imputer = SimpleImputer(strategy='median')
    imputed_train = simple_imputer.fit_transform(working_train_df)
    imputed_val = simple_imputer.transform(working_val_df)
    imputed_test = simple_imputer.transform(working_test_df)

    working_train_df = pd.DataFrame(imputed_train, columns=working_train_df.columns, index=working_train_df.index)
    working_val_df = pd.DataFrame(imputed_val, columns=working_val_df.columns, index=working_val_df.index)
    working_test_df = pd.DataFrame(imputed_test, columns=working_test_df.columns, index=working_test_df.index)


    



    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.
    minmax_scaler = MinMaxScaler()
    scaled_train = minmax_scaler.fit_transform(working_train_df)
    scaled_val = minmax_scaler.transform(working_val_df)
    scaled_test = minmax_scaler.transform(working_test_df)

    working_train_df = pd.DataFrame(scaled_train, columns=working_train_df.columns, index=working_train_df.index)
    working_val_df = pd.DataFrame(scaled_val, columns=working_val_df.columns, index=working_val_df.index)
    working_test_df = pd.DataFrame(scaled_test, columns=working_test_df.columns, index=working_test_df.index)



    return working_train_df.values, working_val_df.values, working_test_df.values


