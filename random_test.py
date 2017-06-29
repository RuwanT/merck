from nutsflow import *
from nutsml import *
import matplotlib.pyplot as plt
from custom_networks import deep_net, merck_net
from custom_metric import Rsqured
import numpy as np
import pandas as pd
from keras.optimizers import Adam, sgd
import sys
from keras.models import model_from_json

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


# if __name__ == "__main__":
#     for dataset_name in dataset_names:
#         print 'Training on Data-set: ' + dataset_name
#         print 'Training on Data-set: ' + dataset_name
#         test_file = data_root + dataset_name + '_test_disguised.csv'
#         train_file = data_root + dataset_name + '_training_disguised.csv'
#
#         data_train = ReadPandas(train_file, dropnan=True)
#         Act_inx = data_train.dataframe.columns.get_loc('Act')
#         feature_dim = data_train.dataframe.shape[1] - (Act_inx + 1)
#
#
#         if NET_ARCH == 'deep_net':
#             model = deep_net(input_shape=(feature_dim,))
#             opti = Adam(lr=0.0001, beta_1=0.5)
#         elif NET_ARCH == 'merck_net':
#             model = merck_net(input_shape=(feature_dim,))
#             opti = sgd(lr=0.05, momentum=0.9, clipnorm=1.0)
#         else:
#             sys.exit("Network not defined correctly, check NET_ARCH. ")
#
#         model.compile(optimizer=opti, loss='mean_squared_error', metrics=[Rsqured])
#         model.load_weights('./outputs/weights_' + dataset_name + '.h5')
#         print model.layers
#
#         for layer in model.layers:
#             if 'dense' in layer.name:
#                 weights = layer.get_weights()
#                 h_0, x_0 = np.histogram(weights[0], 1000, density=True)
#                 # h_1, x_1 = np.histogram(weights[1], 1000, density=True)
#                 print layer.name, weights[0].shape, np.percentile(weights[0],80)
#
#                 plt.plot(x_0[1:], h_0)
#                 # plt.plot(x_1[1:], h_1)
#                 plt.axvline(x=np.percentile(weights[0],80), ymin=0.0, ymax=np.max(h_0), linewidth=2, color='k')
#                 plt.axvline(x=-np.percentile(weights[0], 80), ymin=0.0, ymax=np.max(h_0), linewidth=2, color='k')
#                 plt.show()
#
#
#             else:
#                 print layer.name, 'No weights'


# dataset_name = 'CB1'
# GEN_FEATURE_SELECT = 0
# feature_dim = 10
#
# json_file = open('./outputs/model_' + dataset_name + '_' + str(GEN_FEATURE_SELECT) + '.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# base_model = model_from_json(loaded_model_json)
# # base_model.load_weights('C:\\GIT_codes\\backup_models\\outputs_train_evo_net\\weights_' + dataset_name + '_' + str(GEN_FEATURE_SELECT) + '.h5')
#
# hidden_shape = {'dense_in': feature_dim, 'dense_1': 4000, 'dense_2': 2000, 'dense_3': 1000, 'dense_4': 1000}
# for layer in base_model.layers:
#     if 'dense' in layer.name and 'out' not in layer.name:
#         hidden_shape[layer.name] = layer.get_config()['units']
#         print layer.get_config()['units']
#
# print hidden_shape

# mult = 2.
#
# def test_funct(sample):
#     return sample[0]*mult
#
# [(1,3),(2,7),(2,3),(6,9)] >> Map(test_funct) >> Print() >> Consume()
#
#
# mult = 4.
#
# [(1,3),(2,7),(2,3),(6,9)] >> Map(test_funct) >> Print() >> Consume()


for i in range(1,10):
    fname = '/home/truwan/projects/merck/outputs/w_bm_CB1_0_' + str(i) + '.npy'
    img = np.load(fname).astype(np.float)
    a, x = np.histogram(img, 1000, range=(-.5,1.5), density=True)
    plt.plot(x[:-1], a)
    plt.xlim([-.1, .1])
    plt.pause(1)