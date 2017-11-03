#!/usr/bin/env python

###########################################################################
# data_preprocessing.py: Preprocessing code for sklearn mini-project
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2017-11-03
###########################################################################

import pandas as pd
import sklearn.preprocessing as sk

def read_data(filename):
    """Loads raw data for this project, given a filename; returns
    a Pandas dataframe."""
    # Columns below are copied from
    # https://archive.ics.uci.edu/ml/datasets/adult
    columns = ("age", "workclass", "fnlwgt", "education",
               "education_num", "marital_status", "occupation",
               "relationship", "race", "sex", "capital_gain",
               "capital_loss", "hours_per_week", "native_country",
               "income")
    # Data is CSV with no header; question mark indicates NA
    return pd.read_csv(filename, names=columns, skipinitialspace=True,
                       na_values="?", index_col=False)

def fill_missing(df):
    """Modifies the input dataframe to fill missing data in the columns
    workclass, occupation, and native_country."""
    # Fill the NAs in workclass with the mode ('Private' outnumbers
    # all other categories combined):
    df.workclass.fillna("Private", inplace=True)
    # Do likewise for native_country (vast majority are from US):
    df.native_country.fillna("United-States", inplace=True)
    # NAs in occupation occur primarily where workclass is also NA,
    # but no particular value dominates all the others.  This is still
    # ~6% of our data - so for now, fill it with a new value and treat
    # it perhaps like it has information.
    df.occupation.fillna("Other", inplace=True)

def feature_xform(df):
    """Given raw data (as from 'read_data'), processes all columns into
    numerical form and returns (X, y) where 'X' is a DataFrame for
    features and 'y' is a Series for the corresponding labels (where 0
    is <= 50K, and 1 is > 50K).
    """
    # Extract just the features (everything but 'income'):
    cols = [c for c in df.columns if c != 'income']
    X = df[cols]
    # One-hot encode everything in this tuple, join it to X', and
    # drop the original column:
    onehot_cols = ("workclass", "education", "marital_status", "occupation",
                   "relationship", "race", "native_country")
    for col in onehot_cols:
        feature = X[col]
        feature_onehot = pd.get_dummies(feature, col)
        X = X.join(feature_onehot).drop(col, axis=1)
    # Gender is binary (here at least):
    X = X.assign(male = (X.sex == "Male")*1).drop("sex", axis=1)
    # Encode label, which is also binary:
    y = (df.income == ">50K") * 1
    return (X, y)

def standardize(train, test):
    ss = sk.StandardScaler()
    # Numeric columns needing standardization:
    num_cols = ("age", "fnlwgt", "education_num", "capital_gain",
                "capital_loss", "hours_per_week")
    train.loc[:, num_cols] = ss.fit_transform(train.loc[:,num_cols])
    # Use the same transform on test:
    test.loc[:, num_cols] = ss.transform(test.loc[:,num_cols])

train_raw = read_data("data/train_data.txt")
fill_missing(train_raw)
train_X, train_y = feature_xform(train_raw)

test_raw = read_data("data/test_data.txt")
fill_missing(test_raw)
test_X, test_y =  feature_xform(train_raw)

standardize(train_X, test_X)

# TODO:
# Document 'standardize'
# Perhaps be consistent on whether functions modify dataframe or not

# N.B.:
# The labels are very lopsided (it's 0 around 25% of the time)
# 'education' appears redundant with 'education_num'
# 'relationship' is likely redundant with 'marital_status' (certain ones at least)
# 'race' might have some redundancy with 'native_country'
