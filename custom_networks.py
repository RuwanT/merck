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

    #model.summary()

    return model
