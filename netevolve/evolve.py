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


def init_weight_mask_fs(model, in_layer_name='dense_in'):
    """
    initialize the weight mask used for feature selection to all ones
    :param model: Keras model
    :param flayer_name: name of the first input layer
    :return: Mask matrices for each layer as a dict with layer name as the key. Each mask matrix is a numpy ndarray of 
                ones with shape of layer weights.          
    """
    weight_mask = None
    for layer in model.layers:
        if in_layer_name in layer.name:
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


def sample_weight_mask_fs(model, important_features, Fc=0.8, sampling_type='deterministic', first_hidden='dense_1', correlation_matrix=None):
    for layer in model.layers:
        if first_hidden in layer.name:
            used_features = np.nonzero(important_features)[0].tolist()
            if 'deterministic' in sampling_type:
                if not correlation_matrix is None:
                    m = len(used_features)
                    weights = np.mean(np.abs(layer.get_weights()[0]), axis=1)
                    num_features = int((1. - Fc) * m)
                    cluster_mask = np.zeros(shape=(m,), dtype=np.int8)
                    cluster_ids = range(0, m)
                    for i in range(0, num_features):
                        x = np.random.choice(cluster_ids, size=None, replace=False, p=weights[cluster_ids]/np.sum(weights[cluster_ids]))
                        cluster_mask[x] = 1
                        corr_vect = correlation_matrix[[used_features[i] for i in cluster_ids], used_features[x]]
                        weights[cluster_ids] = weights[cluster_ids] * np.exp(-corr_vect)
                        cluster_ids.remove(x)
                        # weights[x] = 0
                else:
                    weights = np.mean(np.abs(layer.get_weights()[0]), axis=1)
                    fc_percentile = np.percentile(weights, int((1. - Fc)*100.))
                    cluster_mask = (weights > fc_percentile).astype(np.int8)
            elif 'importance' in sampling_type:
                # TODO : not changed to new sampling
                weights = np.mean(np.abs(layer.get_weights()[0]), axis=1)
                weights_ = normalize_cluster_weights(weights) * Fc
                uniform_mat = np.random.random_sample(weights_.shape)
                cluster_mask = (weights_ > uniform_mat).astype(np.int8)

            cols_to_delete = np.nonzero(np.logical_not(cluster_mask).astype(np.int8))
            used_features = np.delete(used_features, cols_to_delete)
            important_features = np.zeros(shape=(len(important_features),), dtype=np.uint8)
            important_features[used_features] = 1

    return important_features


def load_weights_fs(model, base_model, important_features, first_hidden='dense_1'):
    rows_to_delete = np.nonzero(np.logical_not(important_features).astype(np.int8))

    for layer in model.layers:
        if 'dense' in layer.name:
            base_layer = base_model.get_layer(layer.name)
            if first_hidden in layer.name:
                weights = base_layer.get_weights()
                weights[0] = np.delete(weights[0], rows_to_delete, axis=0)
                layer.set_weights(weights)
                layer.trainable = True
            else:
                layer.set_weights(base_layer.get_weights())
                layer.trainable = True

    return model