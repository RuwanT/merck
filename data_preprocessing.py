"""
convert the merck data-set suitable to be fead to the CNN

1) remove columns that does not appear in both training and test
2) normalize the activation to have zero mean and 1 std (z-score)
3) rescale the features to 0-1 by dividing each column by its training max

# TODO: check for features that are not categorical

"""

import pandas as pd
import numpy as np

data_root = '/home/truwan/DATA/merck/'
save_root = '/home/truwan/DATA/merck/preprocessed/'


dataset_names = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1', 'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F', 'TDI', 'THROMBIN']


for dataset_name in dataset_names:

    test_filename = data_root + dataset_name + '_test_disguised.csv'
    train_filename = data_root + dataset_name + '_training_disguised.csv'

    test_filename_save = save_root + dataset_name + '_test_disguised.csv'
    train_filename_save = save_root + dataset_name + '_training_disguised.csv'

    print 'Preprocessing dataset ', dataset_name

    train = pd.read_csv(train_filename)
    test = pd.read_csv(test_filename)

    print len(train.columns.values)
    print len(test.columns.values)

    train_inx_set = set(train.columns.values)
    test_inx_set = set(test.columns.values)

    # remove columns that are not common to both training and test sets
    train_inx = [inx for inx in train.columns.values if inx in set.intersection(train_inx_set, test_inx_set)]
    test_inx = [inx for inx in test.columns.values if inx in set.intersection(train_inx_set, test_inx_set)]

    # print train_inx
    # print test_inx

    train = train[train_inx]
    test = test[test_inx]

    print train.shape
    print test.shape

    # Normalize activations
    X = np.asarray(train.Act)
    x_mean = np.mean(X)
    x_std = np.std(X)

    train.Act = (train.Act - x_mean) / x_std
    test.Act = (test.Act - x_mean) / x_std

    # rescale features
    max_feature = train.max(axis=0)[2:]

    train.ix[:, 2:] = train.ix[:, 2:] / max_feature
    test.ix[:, 2:] = test.ix[:, 2:] / max_feature

    # print train.max(axis=0)

    # save data to csv
    train.to_csv(train_filename_save)
    test.to_csv(test_filename_save)

    print 'Done dataset ', dataset_name
