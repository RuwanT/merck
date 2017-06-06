import numpy as np


# TODO: use sparse layer instead of zero weights to disable synapses

def init_weight_mask(model):
    """
    initialize the weight mask used for evolution to all ones
    :param model: Keras model
    :return: Mask matrices for each layer as a dict with layer name as the key. Each mask matrix is a numpy ndarray of 
                ones with shape of layer weights.          
    """

    weight_mask = dict()
    for layer in model.layers:
        if 'dense' in layer.name and '_out' not in layer.name:
            weight_shape = layer.get_weights()[0].shape
            weight_mask_temp = np.ones(weight_shape, dtype=np.int8)
            weight_mask[layer.name] = weight_mask_temp

    return weight_mask


def init_weight_mask_fs(model, flayer_name='dense_in'):
    """
    initialize the weight mask used for feature selection to all ones
    :param model: Keras model
    :param flayer_name: name of the first input layer
    :return: Mask matrices for each layer as a dict with layer name as the key. Each mask matrix is a numpy ndarray of 
                ones with shape of layer weights.          
    """
    weight_mask = None
    for layer in model.layers:
        if flayer_name in layer.name:
            weight_shape = layer.get_weights()[0].shape
            weight_mask = np.eye(weight_shape[0], weight_shape[1], dtype=np.int8)

    return weight_mask


def evolve_network(model, weight_mask):
    """
    Set the weights of the model to zero when mask is zero. No learning will happen on these synapses.
    :param model: Keras model
    :param weight_mask: dict of numpy matrices with layer name as key. A synapse exists in net gen network if mask==1 
    :return: model with zero weighs where the weight mask is zero
    """

    for layer in model.layers:
        if 'dense' in layer.name and '_out' not in layer.name:
            weights = layer.get_weights()
            weights[0] = (weights[0] * weight_mask[layer.name]).astype(np.float32)
            layer.set_weights(weights)

    return model


def evolve_network_fs(model, weight_mask, flayer_name='dense_in'):
    """
    Set the weights of the model to zero when mask is zero. No learning will happen on these synapses.
    :param model: Keras model
    :param weight_mask: dict of numpy matrices with layer name as key. A synapse exists in net gen network if mask==1 
    :return: model with zero weighs where the weight mask is zero
    """

    for layer in model.layers:
        if flayer_name in layer.name:
            weights = layer.get_weights()
            weights[0] = (weights[0] * weight_mask).astype(np.float32)
            layer.set_weights(weights)

    return model


def normalize_weights(weights, percent=80):
    weights_ = np.abs(weights)
    q80 = np.percentile(weights_, percent)
    weights_ = np.clip(weights_, 0., q80)
    weights_ = np.exp(weights_ / q80 - 1.)

    return weights_


def normalize_cluster_weights(weights, percent=80):
    weights_ = np.abs(weights)
    q80 = np.percentile(weights_, percent)
    weights_ = np.clip(weights_, 0., q80)
    weights_ = np.exp(weights_ / q80 - 1.)

    return weights_


def sample_weight_mask(model, weight_mask, Fs=0.8, Fc=0.8, first_hidden='dense_1'):
    """
    sample the clusters and synapses that would be active in the next genration network    
    :param model: Keras model with trained weights
    :param weight_mask: weight mask from previous generation
    :param Fs: environment factor model for synapses
    :param Fc: environment factor model for clusters
    :param first_hidden: name of the first hidden layer
    :return: weight_mask - updated weight mask, H_shape - num neurones in each layer of next gen network
    """
    H_shape = dict()
    cluster_mask = dict()
    for layer in model.layers:
        if 'dense' in layer.name and '_out' not in layer.name:
            # sample synapses
            weights = layer.get_weights()[0]
            weights_ = normalize_weights(weights) * Fs
            Uniform_mat = np.random.random_sample(weights_.shape)
            weight_mask[layer.name] = np.logical_and((weights_ > Uniform_mat).astype(np.int8),
                                                     weight_mask[layer.name]).astype(np.int8)

            # sample clusters
            weights = np.mean(np.abs(layer.get_weights()[0]), axis=0)
            weights_ = normalize_cluster_weights(weights) * Fc
            Uniform_mat = np.random.random_sample(weights_.shape)
            cluster_mask[layer.name] = (weights_ > Uniform_mat).astype(np.int8)

            H_shape[layer.name] = np.sum(cluster_mask[layer.name])

    for layer in model.layers:
        if 'dense' in layer.name and '_out' not in layer.name:
            if first_hidden in layer.name:
                cols_to_delete = np.nonzero(np.logical_not(cluster_mask[layer.name]).astype(np.int8))
                weight_mask[layer.name] = np.delete(weight_mask[layer.name], cols_to_delete, axis=1)
            else:
                cols_to_delete = np.nonzero(np.logical_not(cluster_mask[layer.name]).astype(np.int8))
                pre_layer_name = 'dense_' + str(int(layer.name.split('_')[1]) - 1)
                rows_to_delete = np.nonzero(np.logical_not(cluster_mask[pre_layer_name]).astype(np.int8))
                weight_mask[layer.name] = np.delete(weight_mask[layer.name], cols_to_delete, axis=1)
                weight_mask[layer.name] = np.delete(weight_mask[layer.name], rows_to_delete, axis=0)

    return weight_mask, H_shape


def sample_weight_mask_fs(model, weight_mask, H_shape,  Fc=0.8, flayer_name='dense_in'):
    for layer in model.layers:
        if flayer_name in layer.name:
            # sample clusters
            weights = np.mean(np.abs(layer.get_weights()[0]), axis=0)
            weights_ = normalize_cluster_weights(weights) * Fc
            Uniform_mat = np.random.random_sample(weights_.shape)
            cluster_mask = (weights_ > Uniform_mat).astype(np.int8)

            cols_to_delete = np.nonzero(np.logical_not(cluster_mask).astype(np.int8))
            weight_mask = np.delete(weight_mask, cols_to_delete, axis=1)
            H_shape[flayer_name] = np.sum(cluster_mask)

    return weight_mask, H_shape