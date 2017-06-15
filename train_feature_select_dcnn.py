from nutsflow import *
from nutsml import *
import matplotlib.pyplot as plt
from custom_networks import deep_net, merck_net, merck_net_fs
from custom_metric import Rsqured
import numpy as np
import pandas as pd
from keras.optimizers import Adam, SGD
import sys
from netevolve import evolve
import os
from keras.models import model_from_json
from keras import backend as K
from multigpu import multi_gpu

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Global variables
BATCH_SIZE = 128
EPOCH = 50
VAL_FREQ = 5
NET_ARCH = 'merck_net_fs'
MAX_GENERATIONS = 10
N_RUNS = 3
GEN_FEATURE_SELECT = 0

data_root = '/home/truwan/DATA/merck/preprocessed/'

dataset_names = ['CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'METAB', 'NK1', 'OX1', 'PGP', 'PPB', 'RAT_F',
                 'TDI', 'THROMBIN', 'OX2', '3A4', 'LOGD']

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
        opti = SGD(lr=0.05, momentum=0.9, clipnorm=1.0)
    elif NET_ARCH == 'merck_net_fs':
        model = merck_net_fs(input_shape=(feature_dim,), hidden_shape=H_shape)
        opti = SGD(lr=0.05, momentum=0.9, clipnorm=1.0)
        # for layer in model.layers:
        #     if 'dense_in' in layer.name:
        #         layer.trainable = False

    else:
        sys.exit("Network not defined correctly, check NET_ARCH. ")

    # model = multi_gpu.make_parallel(model, 2)
    model.compile(optimizer=opti, loss='mean_squared_error', metrics=[Rsqured])
    model.summary()

    return model, opti


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


def is_structure_valid(hidden_shape, min_neurones=5):
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
        print dataset_name + ' has ' + str(feature_dim) + ' features'

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

        trues_val = data_val >> GetCols(Act_inx) >> Map(scale_activators) >> Collect()
        trues_test = data_test >> GetCols(Act_inx) >> Map(scale_activators) >> Collect()
        for rrun in range(0, N_RUNS):
            # TODO : Check code
            json_file = open('./outputs/model_' + dataset_name + '_' + str(GEN_FEATURE_SELECT) + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            base_model = model_from_json(loaded_model_json)
            base_model.load_weights('./outputs/weights_' + dataset_name + '_' + str(GEN_FEATURE_SELECT) + '.h5')

            hidden_shape = {'dense_in': feature_dim, 'dense_1': 4000, 'dense_2': 2000, 'dense_3': 1000, 'dense_4': 1000}
            for layer in base_model.layers:
                if 'dense' in layer.name and 'out' not in layer.name:
                    hidden_shape[layer.name] = layer.get_config()['units']

            model, opti = initialize_model(feature_dim=feature_dim, H_shape=hidden_shape)
            weight_mask = evolve.init_weight_mask_fs(model)
            # make sure the input layer weight matrix has one synapse per neurone
            model = evolve.evolve_network_fs(model, weight_mask)

            for layer in model.layers:
                if 'dense' in layer.name and 'in' not in layer.name:
                    base_layer = base_model.get_layer(layer.name)
                    model.set_weights(base_layer.get_weights())
                    layer.trainable = False


            for gen in range(0, MAX_GENERATIONS):
                print 'Feature Selection dataset ' + dataset_name + ', generation: ' + str(gen)

                best_RMSE = float("inf")
                for e in range(1, EPOCH + 1):
                    # training the network
                    data_train >> Shuffle(1000) >> Map(organize_features) >> NOP(PrintColType()) >> build_batch >> Map(
                        train_network_batch) >> NOP(Print()) >> Consume()

                    # test the network every VAL_FREQ iteration
                    if int(e) % VAL_FREQ == 0:
                        preds = data_val >> Map(organize_features) >> build_batch >> Map(
                            predict_network_batch) >> Flatten() >> Map(scale_activators) >> Collect()

                        RMSE_e = RMSE_np(preds, trues_val)

                        # preds = data_test >> Map(organize_features) >> build_batch >> Map(
                        #     predict_network_batch) >> Flatten() >> Map(scale_activators) >> Collect()
                        #
                        # RMSE_test = RMSE_np(preds, trues_test)
                        #
                        # print e, RMSE_e, RMSE_test, K.get_value(opti.lr)

                        if RMSE_e < best_RMSE:
                            model_json = model.to_json()
                            model.save_weights(
                                './outputs/weights_' + dataset_name + '_' + 'temp1' + '.h5')
                            best_RMSE = RMSE_e

                    # change leaning rate every 10-th epoch
                    if int(e) % 10 == 0:
                        K.set_value(opti.lr, 0.5 * K.get_value(opti.lr))

                # load best model
                model.load_weights('./outputs/weights_' + dataset_name + '_' + 'temp1' + '.h5')

                print "Calculating errors for test set ..."
                preds = data_test >> Map(organize_features) >> build_batch >> Map(
                    predict_network_batch) >> Flatten() >> Map(
                    scale_activators) >> Collect()

                RMSE_e = RMSE_np(preds, trues_test)
                Rsquared_e = Rsqured_np(preds, trues_test)
                print 'Dataset ' + dataset_name + ', Gen ' + str(gen) + ' Test : RMSE = ' + str(
                    RMSE_e) + ', R-Squared = ' + str(Rsquared_e)
                test_stat_hold.append(
                    (rrun, gen, RMSE_e, Rsquared_e, best_RMSE, hidden_shape['dense_in']))

                # model_json = model.to_json()
                # with open('./outputs/model_' + dataset_name + '_' + str(gen) + '_' + str(rrun) + '.json', "w") as json_file:
                #     json_file.write(model_json)
                # model.save_weights('./outputs/weights_' + dataset_name + '_' + str(gen) + '_' + str(rrun) + '.h5')

                # Select features for next generation
                fc = 10 ** ((1. - np.log10(feature_dim)) / 9)
                # TODO : change code so that freezing upper layers is possible
                weight_mask, hidden_shape = evolve.sample_weight_mask_fs(model, weight_mask, hidden_shape, Fc=fc,
                                                                         sampling_type='deterministic')
                if not is_structure_valid(hidden_shape):
                    print 'Stopping Evolution: At least one layer has less than minimum neurones'
                    break

                K.clear_session()
                model, opti = initialize_model(feature_dim=feature_dim, H_shape=hidden_shape)
                model = evolve.evolve_network_fs(model, weight_mask)

                # Save the weight matrix
                selected_features = np.sum(weight_mask, axis=1)
                np.save('./outputs/featureSelect_' + dataset_name + '_' + str(rrun) + '_' + str(gen) + '.npy',
                        selected_features)

        writer = WriteCSV('./outputs/feature_selection_' + dataset_name + '.csv')
        test_stat_hold >> writer
