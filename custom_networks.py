def deep_net(input_shape=(128)):
    from keras import models
    from keras.layers import Dense
    from keras.layers.advanced_activations import LeakyReLU
    from keras.layers.normalization import BatchNormalization
    from keras.layers import Dropout
    from keras.layers.noise import GaussianNoise

    model = models.Sequential()

    model.add(Dense(1024, activation=None, use_bias=False, input_dim=input_shape))
    # model.add(GaussianNoise(0.001))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones',
                                 moving_mean_initializer='zeros',
                                 moving_variance_initializer='ones', beta_regularizer=None,
                                 gamma_regularizer=None, beta_constraint=None,
                                 gamma_constraint=None))
    model.add(Dropout(0.5))

    model.add(Dense(512, activation=None, use_bias=False))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones',
                                 moving_mean_initializer='zeros',
                                 moving_variance_initializer='ones', beta_regularizer=None,
                                 gamma_regularizer=None, beta_constraint=None,
                                 gamma_constraint=None))
    model.add(Dropout(0.5))

    model.add(Dense(128, activation=None, use_bias=False))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones',
                                 moving_mean_initializer='zeros',
                                 moving_variance_initializer='ones', beta_regularizer=None,
                                 gamma_regularizer=None, beta_constraint=None,
                                 gamma_constraint=None))

    model.add(Dropout(0.5))

    model.add(Dense(1, activation=None, use_bias=True))

    # model.summary()

    return model


def merck_net(input_shape=(128), hidden_shape={'dense_1': 4000, 'dense_2': 2000, 'dense_3': 1000, 'dense_4': 1000}):
    """
    # The recommended network presented in the paper: Junshui Ma et. al., Deep Neural Nets as a Method for Quantitative 
    # Structure Activity Relationships
    # URL: http://www.cs.toronto.edu/~gdahl/papers/deepQSARJChemInfModel2015.pdf
    # :param input_shape: dim of input features
    # :return: a keras model
    """

    from keras import models
    from keras.layers import Dense
    from keras.layers.advanced_activations import LeakyReLU
    from keras.layers.normalization import BatchNormalization
    from keras.layers import Dropout
    from keras.layers.noise import GaussianNoise
    from keras.regularizers import l2

    # TODO: is kernel_regularizer=l2(0.0001) the best way to add weight cost strength?
    # TODO: add batch normalization

    model = models.Sequential()

    model.add(Dense(hidden_shape['dense_1'], activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.0001),
                    name='dense_1'))
    model.add(Dropout(0.25, name='drop_1'))

    model.add(Dense(hidden_shape['dense_2'], activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.0001),
                    name='dense_2'))
    model.add(Dropout(0.25, name='drop_2'))

    model.add(Dense(hidden_shape['dense_3'], activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.0001),
                    name='dense_3'))
    model.add(Dropout(0.25, name='drop_3'))

    model.add(Dense(hidden_shape['dense_4'], activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.0001),
                    name='dense_4'))
    model.add(Dropout(0.10, name='drop_4'))

    model.add(Dense(1, activation=None, use_bias=True, kernel_regularizer=l2(0.0001), name='dense_out'))

    # model.summary()

    return model


def merck_net_fs(input_shape=(128),
                 hidden_shape={'dense_in': 4000, 'dense_1': 4000, 'dense_2': 2000, 'dense_3': 1000, 'dense_4': 1000}):
    """
    # The recommended network presented in the paper: Junshui Ma et. al., Deep Neural Nets as a Method for Quantitative 
    # Structure Activity Relationships
    # URL: http://www.cs.toronto.edu/~gdahl/papers/deepQSARJChemInfModel2015.pdf
    # :param input_shape: dim of input features
    # :return: a keras model
    """

    from keras import models
    from keras.layers import Dense
    from keras.layers.advanced_activations import LeakyReLU
    from keras.layers.normalization import BatchNormalization
    from keras.layers import Dropout
    from keras.layers.noise import GaussianNoise
    from keras.regularizers import l2
    from keras.initializers import Ones

    # TODO: is kernel_regularizer=l2(0.0001) the best way to add weight cost strength?
    # TODO: add batch normalization

    model = models.Sequential()

    model.add(Dense(hidden_shape['dense_in'], activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.0001),
                    kernel_initializer=Ones(), name='dense_in'))
    # model.add(Dropout(0.25, name='drop_1'))

    model.add(Dense(hidden_shape['dense_1'], activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.0001),
                    name='dense_1'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Dropout(0.25, name='drop_1'))

    model.add(Dense(hidden_shape['dense_2'], activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.0001),
                    name='dense_2'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Dropout(0.25, name='drop_2'))

    model.add(Dense(hidden_shape['dense_3'], activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.0001),
                    name='dense_3'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Dropout(0.25, name='drop_3'))

    model.add(Dense(hidden_shape['dense_4'], activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.0001),
                    name='dense_4'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Dropout(0.10, name='drop_4'))

    model.add(Dense(1, activation=None, use_bias=True, kernel_regularizer=l2(0.0001), name='dense_out'))

    # model.summary()

    return model
