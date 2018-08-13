import numpy as np
import tensorflow as tf

def batch_means_ess(x):
    """ Estimate the effective sample size as the ratio of the variance
    of the batch means to the variance of the chain. As explained here:
    https://arxiv.org/pdf/1011.0175.pdf. We expect the chain in the format
    Time-Steps, Num-Chains, Dimension (T, M, D) """

    T, M, D = x.shape
    num_batches = int(np.floor(T**(1/3)))
    batch_size = int(np.floor(num_batches**2))
    batch_means = []
    for i in range(num_batches):
        batch = x[batch_size*i:batch_size*i + batch_size]
        batch_means.append(np.mean(batch, axis=0))
    batch_variance = np.var(np.array(batch_means), axis=0)
    chain_variance = np.var(x, axis=0)

    act = batch_size * batch_variance/(chain_variance + 1e-20)

    return 1/act


def batch_means_essTF(x):
    """ Tensorflow calcualtion of ESS so that we can monitor this
    quantity during training"""

    T, M, D = x.shape
    num_batches = tf.floor(tf.pow(tf.cast(T, tf.float32), (1 / 3)))
    batch_size = tf.floor(num_batches ** 2)
    x = x[:tf.cast(num_batches * batch_size, tf.int32)]
    batches = tf.stack(tf.split(x, tf.cast(num_batches, tf.float32)))
    batch_means = tf.reduce_mean(batches, axis=0)
    _, batch_variance = tf.nn.moments(tf.stack(batch_means), axis=0)
    _, chain_variance = tf.nn.moments(x, axis=0)

    act = batch_size * batch_variance / (chain_variance + 1e-20)

    return 1 / act
