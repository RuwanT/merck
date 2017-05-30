from nutsflow import *
from nutsml import *
import matplotlib.pyplot as plt
from custom_networks import deep_net, merck_net
from custom_metric import Rsqured
import numpy as np
import pandas as pd
from keras.optimizers import Adam, sgd
import sys


# Global variables
BATCH_SIZE = 64
EPOCH = 200
VAL_FREQ = 5
NET_ARCH = 'merck_net'
data_root = '/home/truwan/DATA/merck/preprocessed/'
dataset_names = ['CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'METAB', 'NK1', 'OX1', 'PGP', 'PPB', 'RAT_F',
                 'TDI', 'THROMBIN', 'OX2', '3A4', 'LOGD']
dataset_names = ['CB1', 'DPP4']

dataset_stats = pd.read_csv(data_root + 'dataset_stats.csv', header=None, names=['mean', 'std'], index_col=0)


if __name__ == "__main__":
    for dataset_name in dataset_names:
        print 'Training on Data-set: ' + dataset_name
        print 'Training on Data-set: ' + dataset_name
        test_file = data_root + dataset_name + '_test_disguised.csv'
        train_file = data_root + dataset_name + '_training_disguised.csv'

        data_train = ReadPandas(train_file, dropnan=True)
        Act_inx = data_train.dataframe.columns.get_loc('Act')
        feature_dim = data_train.dataframe.shape[1] - (Act_inx + 1)


        if NET_ARCH == 'deep_net':
            model = deep_net(input_shape=(feature_dim,))
            opti = Adam(lr=0.0001, beta_1=0.5)
        elif NET_ARCH == 'merck_net':
            model = merck_net(input_shape=(feature_dim,))
            opti = sgd(lr=0.05, momentum=0.9, clipnorm=1.0)
        else:
            sys.exit("Network not defined correctly, check NET_ARCH. ")

        model.compile(optimizer=opti, loss='mean_squared_error', metrics=[Rsqured])
        model.load_weights('./outputs/weights_' + dataset_name + '.h5')
        print model.layers

        for layer in model.layers:
            if 'dense' in layer.name:
                weights = layer.get_weights()
                h_0, x_0 = np.histogram(weights[0], 1000, density=True)
                # h_1, x_1 = np.histogram(weights[1], 1000, density=True)
                print layer.name, weights[0].shape, np.percentile(weights[0],80)

                plt.plot(x_0[1:], h_0)
                # plt.plot(x_1[1:], h_1)
                plt.axvline(x=np.percentile(weights[0],80), ymin=0.0, ymax=np.max(h_0), linewidth=2, color='k')
                plt.axvline(x=-np.percentile(weights[0], 80), ymin=0.0, ymax=np.max(h_0), linewidth=2, color='k')
                plt.show()


            else:
                print layer.name, 'No weights'