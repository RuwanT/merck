import keras
from nutsflow import *
from nutsml import *
import matplotlib.pyplot as plt
from custom_networks import deep_net
from custom_metric import Rsqured
import numpy as np

BATCH_SIZE = 128
EPOCH = 200
dataset_num = 4

train_file = '/home/truwan/DATA/TrainingSet/ACT' + str(dataset_num) + '_competition_training.csv'
test_file = '/home/truwan/DATA/TestSet/ACT' + str(dataset_num) + '_competition_test.csv'


def Rsqured_np(x,y):

    x = np.asarray(x)
    print x.shape
    y = np.asarray(y)
    print y.shape

    avx = np.mean(x)
    avy = np.mean(y)

    num = np.sum((x-avx) * (y-avy))
    num = num * num

    denom = np.sum((x-avx)*(x-avx)) * np.sum((y-avy)*(y-avy))

    return num/denom

if __name__ == "__main__":
    print "Starting the code ..."

    data_train = ReadPandas(train_file, dropnan=True)
    feature_dim = data_train.dataframe.shape[1] - 2


    def organize_features_train(sample):
        """
        reorganize the flow as a feature vector predictor pair
        :param sample: A row of data comming through the pipe
        :return: a tupe consising feature vector and predictor
        """
        y = [sample[1], ]
        features = list(sample[2:])
        return (features, y)


    build_batch = (BuildBatch(BATCH_SIZE)
                   .by(0, 'vector', float)
                   .by(1, 'number', float))

    model = deep_net(input_shape=feature_dim)
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=[Rsqured])


    def train_network_batch(sample):
        tloss = model.train_on_batch(sample[0], sample[1])
        return (tloss[0], tloss[1])


    for e in range(1, EPOCH):
        data_train >> Shuffle(1000) >> Map(organize_features_train) >> build_batch >> Map(
            train_network_batch) >> Print() >> Consume()
        print "Epoch ", str(e) , "done"

    print "Calculating final R2 error ..."

    def predict_network_batch(sample):
        return model.predict(sample[0])

    preds = data_train >> Map(organize_features_train) >> build_batch >> Map(predict_network_batch) >> Flatten() >> Collect()
    trues = data_train >> GetCols(1) >> Collect()

    print "Final batchwise Rsquared error average: ", Rsqured_np(preds, trues)
