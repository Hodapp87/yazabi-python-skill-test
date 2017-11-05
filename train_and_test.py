#!/usr/bin/env python

###########################################################################
# train_and_test.py: Training & validation for sklearn mini-project
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2017-11-03
###########################################################################

import data_preprocessing

import pandas as pd
import sklearn

def train_and_validate(algorithm):
    """This function is mandated by the requirements.  Argument
    'algorithm' takes values of 'naive_bayes', 'decision_tree', 'knn',
    and 'svm', and the function will train the respective classifier
    on training data, predict on the testing data, and print that
    classifier's *testing* accuracy.
    """
    # I'm going to assume we should use the specified classifier by
    # itself, and using it as the base classifier in an ensemble is
    # cheating.
    train_X, train_y, test_X, test_y = data_preprocessing.get_processed_data()

# N.B.:
# The labels are very lopsided (it's 0 around 25% of the time)
# 'education' appears redundant with 'education_num'
# 'relationship' is likely redundant with 'marital_status' (certain ones at least)
# 'race' might have some redundancy with 'native_country'
