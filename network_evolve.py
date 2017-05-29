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
            weights = layer.get_weights()
            if len(weights) > 0:
                print layer.name, 'weight dimention: ', weights[0].shape, np.mean(weights[0]), np.max(weights[0]), np.min(weights[0])

                # layer.set_weights([np.zeros(weights[0].shape, dtype=np.float32), np.zeros(weights[1].shape, np.float32)])
                # weights = layer.get_weights()
                h,x = np.histogram(weights[0][:, 0:10], 1000, density=True)
                # plt.plot(x[1:],h)
                plt.imshow(weights[0])
                plt.show()

            else:
                print layer.name, 'No weights'