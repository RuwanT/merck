import numpy as np


def init_weight_mask(model):
    """
    initialize the weight mask used for evolution
    :param model: network structure
    :return: dict of the mask matrix with all ones with same dimension as layer weights
                
    """

    # TODO: Write docstring
    weight_mask = dict()
    for layer in model.layers:
        if 'dense' in layer.name and '_out' not in layer.name:
            weight_shape = layer.get_weights()[0].shape
            weight_mask_temp = np.ones(weight_shape, dtype=np.int8)
            weight_mask[layer.name] = weight_mask_temp

    return weight_mask


def evolve_network(model, weight_mask):
    """
    
    :param model: model with wegits initialized by gaussian
    :param weight_mask: list of numpy matrices indicating which weight will be non zero in the next generation
    :return: model with zero weighs where the weight mask is zero
    """
    # TODO: Write docstring

    for layer in model.layers:
        if 'dense' in layer.name and '_out' not in layer.name:
            weights = layer.get_weights()
            weights[0] = (weights[0]*weight_mask).astype(np.float32)
            layer.set_weight(weights)

    return model


def normalize_weights(weights, percent=80):
    weights_ = np.abs(weights)
    q80 = np.percentile(weights_, percent)
    weights_ = np.clip(weights_, 0., q80)
    weights_ = weights_ / q80

    return weights_


def normalize_cluster_weights(weights, percent=80):
    weights_ = np.abs(weights)
    q80 = np.percentile(weights_, percent)
    weights_ = np.clip(weights_, 0., q80)
    weights_ = weights_ / q80

    return weights_


def sample_weight_mask(model, weight_mask, Fs=0.8, Fc=0.8, first_hidden='dense_1'):
    """
    
    :param model: 
    :param weight_mask: 
    :param cluster_mask: 
    :return: 
    """
    H_shape = dict()
    cluster_mask = dict()
    for layer in model.layers:
        if 'dense' in layer.name and '_out' not in layer.name:

            # sample synapses
            weights = layer.get_weights()[0]
            weights_ = normalize_weights(weights)*Fs
            Uniform_mat = np.random.random_sample(weights.shape)
            weight_mask[layer.name] = np.logical_and((weights_ > Uniform_mat).astype(np.int8), weight_mask[layer.name]).astype(np.int8)

            # sample clusters
            weights = np.mean(np.abs(layer.get_weights()[0]), axis=0)
            weights_ = normalize_cluster_weights(weights)*Fc
            Uniform_mat = np.random.random_sample(weights.shape)
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
