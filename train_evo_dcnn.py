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
import os
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"]="5,6,7,8"

# Global variables
BATCH_SIZE = 64
EPOCH = 50
VAL_FREQ = 5
NET_ARCH = 'merck_net'
MAX_GENERATIONS = 10
N_RUNS = 10

data_root = '/home/truwan/DATA/merck/preprocessed/'

dataset_names = ['CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'METAB', 'NK1', 'OX1', 'PGP', 'PPB', 'RAT_F',
                 'TDI', 'THROMBIN', 'OX2', '3A4', 'LOGD']
dataset_names = ['CB1', 'LOGD']

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


def is_structure_valid(hidden_shape, min_neurones=10):
    is_valid = True
    for k, h in hidden_shape.iteritems():
        if h < min_neurones:
            is_valid = False
    return is_valid


def count_trainable_parameters(model, weight_mask):
    num_parameters = 0.
    for layer in model.layers:
        if 'dense' in layer.name:
            bias_shape = layer.get_weights()[1].shape[0]

            if layer.name in weight_mask.keys():
                num_parameters = num_parameters + np.sum(weight_mask[layer.name]) + bias_shape
            else:
                weights_shape = layer.get_weights()[0].shape
                num_parameters += int(weights_shape[0]) + bias_shape
    return num_parameters


def get_stats(error_list, gen):
    mean_rmse = np.mean([rrun_stat[3] for rrun_stat in error_list if rrun_stat[1] == gen])
    std_rmse = np.std([rrun_stat[3] for rrun_stat in error_list if rrun_stat[1] == gen])
    med_rmse = np.median([rrun_stat[3] for rrun_stat in error_list if rrun_stat[1] == gen])

    return mean_rmse, std_rmse, med_rmse


if __name__ == "__main__":
    for dataset_name in dataset_names:
        test_stat_hold = list()

        print 'Training on Data-set: ' + dataset_name
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

        for gen in range(0, MAX_GENERATIONS):
            best_RMSE_gen = float("inf")
            for rrun in range(0, N_RUNS):
                trues = data_val >> GetCols(Act_inx) >> Map(scale_activators) >> Collect()
                best_RMSE = float("inf")
                print 'Training dataset ' + dataset_name + ', generation: ' + str(gen)
                for e in range(1, EPOCH + 1):
                    # training the network
                    data_train >> Shuffle(1000) >> Map(organize_features) >> NOP(PrintColType()) >> build_batch >> Map(
                        train_network_batch) >> NOP(Print()) >> Consume()

                    # test the network every VAL_FREQ iteration
                    if int(e) % VAL_FREQ == 0:
                        preds = data_val >> Map(organize_features) >> build_batch >> Map(
                            predict_network_batch) >> Flatten() >> Map(scale_activators) >> Collect()

                        RMSE_e = RMSE_np(preds, trues)
                        # Rsquared_e = Rsqured_np(preds, trues)

                        if RMSE_e < best_RMSE:
                            model_json = model.to_json()
                            # with open('./outputs/model_' + dataset_name + '_' + str(gen) + '.json', "w") as json_file:
                            # json_file.write(model_json)
                            model.save_weights(
                                './outputs/weights_' + dataset_name + '_' + str(gen) + '_' + 'temp' + '.h5')
                            best_RMSE = RMSE_e

                # loading the model based on best validation error
                model.load_weights('./outputs/weights_' + dataset_name + '_' + str(gen) + '_' + 'temp' + '.h5')
                print "Calculating errors for validation set ..."
                preds = data_val >> Map(organize_features) >> build_batch >> Map(
                    predict_network_batch) >> Flatten() >> Map(scale_activators) >> Collect()
                RMSE_e = RMSE_np(preds, trues)
                if RMSE_e < best_RMSE_gen:
                    model_json = model.to_json()
                    with open('./outputs/model_' + dataset_name + '_' + str(gen) + '.json', "w") as json_file:
                        json_file.write(model_json)
                    model.save_weights('./outputs/weights_' + dataset_name + '_' + str(gen) + '.h5')
                    best_RMSE_gen = RMSE_e

                print "Calculating errors for test set ..."
                trues = data_test >> GetCols(Act_inx) >> Map(scale_activators) >> Collect()

                preds = data_test >> Map(organize_features) >> build_batch >> Map(
                    predict_network_batch) >> Flatten() >> Map(
                    scale_activators) >> Collect()
                RMSE_e_val = RMSE_e
                RMSE_e = RMSE_np(preds, trues)
                Rsquared_e = Rsqured_np(preds, trues)
                print 'Dataset ' + dataset_name + ', Gen ' + str(gen) + ' Test : RMSE = ' + str(
                    RMSE_e) + ', R-Squared = ' + str(Rsquared_e)
                test_stat_hold.append(('Gen_' + str(gen), gen, rrun, RMSE_e, Rsquared_e, RMSE_e_val))

            # Evolve network
            # TODO: check if making the weights to zero works
            model.load_weights('./outputs/weights_' + dataset_name + '_' + str(gen) + '.h5')
            weight_mask, hidden_shape = evolve.sample_weight_mask(model, weight_mask)
            mean_rmse, std_rmse, med_rmse = get_stats(test_stat_hold, gen)
            test_stat_hold.append(('Gen_' + str(gen), 'stat', mean_rmse, std_rmse, med_rmse,
                                   count_trainable_parameters(model, weight_mask)))
            if not is_structure_valid(hidden_shape):
                print 'Stopping Evolution: At least one layer has less than minimum neurones'
                break

            K.clear_session()   #clear the old model from GPU memory
            model = initialize_model(feature_dim=feature_dim, H_shape=hidden_shape)
            model.summary()
            model = evolve.evolve_network(model, weight_mask)

        writer = WriteCSV('./outputs/test_errors_' + dataset_name + '.csv')
        test_stat_hold >> writer
