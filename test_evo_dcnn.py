from nutsflow import *
from nutsml import *
import matplotlib.pyplot as plt
from custom_networks import deep_net, merck_net
from custom_metric import Rsqured
import numpy as np
import pandas as pd
from keras.optimizers import Adam, sgd
import sys
from netevolve import evolve
from keras.models import model_from_json

# Global variables
BATCH_SIZE = 64
EPOCH = 200
VAL_FREQ = 5
NET_ARCH = 'merck_net'
MAX_GENERATIONS = 10
PLOT_FEATURE_IMP = False
PLOT_ACCURACY_GEN = True

data_root = '/home/truwan/DATA/merck/preprocessed/'

dataset_names = ['CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'METAB', 'NK1', 'OX1', 'PGP', 'PPB', 'RAT_F',
                 'TDI', 'THROMBIN', 'OX2', '3A4', 'LOGD']

dataset_names = ['OX1', 'PGP', 'PPB', 'RAT_F',
                 'TDI', 'THROMBIN', 'OX2']

dataset_stats = pd.read_csv(data_root + 'dataset_stats.csv', header=None, names=['mean', 'std'], index_col=0)


def initialize_model(feature_dim, H_shape):
    """
    initialize the keras model
    :param feature_dim: input feature shape
    :param H_shape: dictionary with number of neurones in each layer
    :return: 
    """
    if NET_ARCH == 'deep_net':
        model = deep_net(input_shape=(feature_dim,))
        opti = Adam(lr=0.0001, beta_1=0.5)
    elif NET_ARCH == 'merck_net':
        model = merck_net(input_shape=(feature_dim,), hidden_shape=H_shape)
        opti = sgd(lr=0.05, momentum=0.9, clipnorm=1.0)
    else:
        sys.exit("Network not defined correctly, check NET_ARCH. ")

    model.compile(optimizer=opti, loss='mean_squared_error', metrics=[Rsqured])

    return model


def Rsqured_np(x, y):
    """
    calculates r2 error in numpy
    :param x: true values
    :param y: predicted values
    :return: r2 error
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    avx = np.mean(x)
    avy = np.mean(y)

    num = np.sum((x - avx) * (y - avy))
    num = num * num

    denom = np.sum((x - avx) * (x - avx)) * np.sum((y - avy) * (y - avy))

    return num / denom


def RMSE_np(x, y):
    """
    calculates r2 error in numpy
    :param x: true values
    :param y: predicted values
    :return: RMSE error
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = x.shape[0]

    return np.sqrt(np.sum(np.square(x - y)) / n)


def count_trainable_weights(model):
    num_weights = 0.
    for layer in model.layers:
        if 'dense' in layer.name:
            weights = layer.get_weights()
            num_weights += weights[0].shape[0]*weights[0].shape[1]
            num_weights += weights[1].shape[0]
            # print 'Zero weights: ', np.sum(np.isclose(weights[0], np.zeros(weights[0].shape, dtype=np.float32)).astype(np.int8)), 'Non zero weigths', weights[0].shape[0]*weights[0].shape[1]
            num_weights -= np.sum(np.isclose(weights[0], np.zeros(weights[0].shape, dtype=np.float32)).astype(np.int8))

    return num_weights


def get_feature_importance(model, first_layer='dense_1'):
    """
    Calculate the input feature importance based on the first layer weights
    :param model: Keras model
    :param first_layer: Name of the first layer
    :return: Vector with feature importance (should sum to 1.)
    """

    for layer in model.layers:
        if first_layer in layer.name:
            weights = layer.get_weights()[0]
            weights_ = np.mean(np.abs(weights), axis=1)
            weights_ /= np.max(weights_)

    return weights_


if __name__ == "__main__":
    for dataset_name in dataset_names:
        test_stat_hold = list()

        print 'Testing on Data-set: ' + dataset_name
        test_file = data_root + dataset_name + '_test_disguised.csv'
        train_file = data_root + dataset_name + '_training_disguised.csv'

        data_train = ReadPandas(train_file, dropnan=True)
        Act_inx = data_train.dataframe.columns.get_loc('Act')
        feature_dim = data_train.dataframe.shape[1] - (Act_inx + 1)

        # split randomly train and val
        data_train, data_val = data_train >> SplitRandom(ratio=0.8) >> Collect()
        data_test = ReadPandas(test_file, dropnan=True)


        def organize_features(sample):
            """
            reorganize the flow as a feature vector predictor pair
            :param sample: A row of data comming through the pipe
            :return: a tupe consising feature vector and predictor
            """
            y = [sample[Act_inx], ]
            features = list(sample[Act_inx + 1:])
            return (features, y)


        build_batch = (BuildBatch(BATCH_SIZE)
                       .by(0, 'vector', float)
                       .by(1, 'number', float))

        model = initialize_model(feature_dim=feature_dim,
                                 H_shape={'dense_1': 4000, 'dense_2': 2000, 'dense_3': 1000, 'dense_4': 1000})
        weight_mask = evolve.init_weight_mask(model)


        def train_network_batch(sample):
            tloss = model.train_on_batch(sample[0], sample[1])
            return (tloss[0], tloss[1])


        def test_network_batch(sample):
            tloss = model.test_on_batch(sample[0], sample[1])
            return (tloss[0],)


        def predict_network_batch(sample):
            return model.predict(sample[0])


        scale_activators = lambda x: (
            x[0] * dataset_stats.loc[dataset_name, 'std'] + dataset_stats.loc[dataset_name, 'mean'])

        first_RMSE = 0.
        first_num_trainable_weights = 0.
        if PLOT_ACCURACY_GEN:
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
        for gen in range(1, MAX_GENERATIONS):
            print "Calculating errors for test set ..."
            json_file = open('./outputs/model_' + dataset_name + '_' + str(gen) + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights('./outputs/weights_' + dataset_name + '_' + str(gen) + '.h5')

            trues = data_test >> GetCols(Act_inx) >> Map(scale_activators) >> Collect()

            preds = data_test >> Map(organize_features) >> build_batch >> Map(
                predict_network_batch) >> Flatten() >> Map(
                scale_activators) >> Collect()

            RMSE_e = RMSE_np(preds, trues)
            Rsquared_e = Rsqured_np(preds, trues)

            num_trainable_weights = count_trainable_weights(model)

            feature_importance = get_feature_importance(model)

            print 'Dataset ' + dataset_name + ', Gen ' + str(gen) + ' Test : RMSE = ' + str(
                RMSE_e) + ', R-Squared = ' + str(Rsquared_e) + ', num trainiale weights = ' + str(num_trainable_weights)
            test_stat_hold.append(('Gen_' + str(gen), RMSE_e, Rsquared_e, num_trainable_weights))

            # plot histogram of layer 1 weights per feature
            if PLOT_FEATURE_IMP:
                h, x = np.histogram(feature_importance, bins=255, density=True, range=(0,1))
                bincenters = 0.5 * (x[1:] + x[:-1])
                plt.plot(bincenters, h, label='gen-'+str(gen))
                plt.pause(1.)

            if PLOT_ACCURACY_GEN:
                if gen == 1:
                    first_RMSE = RMSE_e
                    first_num_trainable_weights = num_trainable_weights

                ax1.plot(gen, first_RMSE/RMSE_e, 'ro')
                ax2.plot(gen, first_num_trainable_weights/num_trainable_weights, 'bo' )
                plt.pause(0.1)

        writer = WriteCSV('./outputs/test_errors_' + dataset_name + '.csv')
        test_stat_hold >> writer

        # plot histogram of layer 1 weights per feature
        if PLOT_FEATURE_IMP:
            plt.legend()
            plt.savefig('./outputs/feature_importance_' + dataset_name + '.tiff')
            plt.clf()

        if PLOT_ACCURACY_GEN:
            ax1.set_ylim([0, 2])
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Relative Accuracy', color='r')
            ax1.tick_params('y', colors='r')

            ax2.set_ylim([0, 50])
            ax2.set_ylabel('Relative Efficiency', color='b')
            ax2.tick_params('y', colors='b')
            fig.tight_layout()

            plt.savefig('./outputs/Accuracy_gen_' + dataset_name + '.tiff')
            plt.close()