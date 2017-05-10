import pandas as pd
import numpy as np
from nutsflow import *
from nutsml import *

data_root = '/home/truwan/DATA/merck/'
save_root = '/home/truwan/DATA/merck/preprocessed/'


dataset_names = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1', 'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F', 'TDI', 'THROMBIN']

# for dataset_name in dataset_names:
#     train_filename = data_root + dataset_name + '_training_disguised.csv'
#
#     print 'Preprocessing dataset ', dataset_name
#
#     train = pd.read_csv(train_filename)
#
#     df = train.loc[:, train.dtypes == int]
#     print 'Int columns: ', len(df.columns.values)
#     print 'All columns: ', len(train.columns.values)

dataset_stats = pd.read_csv(save_root + 'dataset_stats.csv', header=None, names=['mean', 'std'], index_col=0)

print dataset_stats
print dataset_stats.loc['DPP4', 'mean']
