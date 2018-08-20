import numpy as np
import tensorflow as tf
from a_nice_mc.objectives import Energy
from a_nice_mc.utils.evaluation import effective_sample_size, acceptance_rate
from a_nice_mc.utils.logger import save_ess, create_logger, ensure_directory




class NN:
    def __init__(self, data, labels, arch, act=tf.nn.tanh, batch_size=None,
                 loc=0.0, prec=1.0):
        """
        Bayesian Neural Net model (assume Normal prior)
        :param data: data for Logistic Regression task
        :param labels: label for Logistic Regression task
        :param batch_size: batch size for Logistic Regression; setting it to None
        adds flexibility at the cost of speed.
        :param loc: mean of the Normal prior
        :param scale: std of the Normal prior
        """
        self.arch = arch
        self.theta_dim = np.sum([arch[i] * arch[i + 1] for i in range(len(arch) - 1)])
        self.act = act
        self.x_dim = data.shape[1]
        self.y_dim = labels.shape[1]
        self.prec_prior = prec

        self.data = tf.constant(data, tf.float32)
        self.labels = tf.constant(labels, tf.float32)
        ensure_directory('logs/net.log')
        self.logger = create_logger(__name__, log_dir="logs/net/", file_name='net.log')

    def _unflatten(self, theta):
        """theta is assumed to have shape (num_chains, target_dim)"""
        m = tf.shape(theta)[0]  # num chains
        weights = []
        start = 0
        for i in range(len(self.arch) - 1):
            size = self.arch[i] * self.arch[i + 1]
            w = tf.reshape(theta[:, start:start + size],
                           (m, self.arch[i], self.arch[i + 1]))
            weights.append(w)
            start += size
        return weights

    def energy_fn(self, theta, x, y):
        """ theta has shape  (num_chains, target_dim)"""
        h = tf.expand_dims(x, 0)
        h = tf.concat([h, tf.ones((1, h.shape[1], 1))], axis=2)
        h = tf.tile(h, [tf.shape(theta)[0], 1, 1])
        weights = self._unflatten(theta)
        for W in weights[:-1]:
            h = self.act(h @ W)
        mean = h @ weights[-1]
        mahalob = 0.5 * tf.reduce_sum((y - mean) ** 2, axis=2)
        prior = 0.5 * tf.reduce_sum(theta ** 2, axis=1, keepdims=True)

        return tf.reduce_sum(mahalob + self.prec_prior * prior, axis=1)

    def __call__(self, v):
        return self.energy_fn(v, self.data, self.labels)

