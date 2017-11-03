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
    pass
