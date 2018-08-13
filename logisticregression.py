import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils.func_utils import accept, jacobian, autocovariance, get_log_likelihood, \
    get_data, binarize, normal_kl, acl_spectrum, ESS
from utils.distributions import Gaussian, GMM, GaussianFunnel, gen_ring, LogisticRegressionTF
from utils.layers import Linear, Sequential, Zip, Parallel, ScaleTanh
from utils.dynamics import Dynamics
from utils.sampler import propose
from utils.notebook_utils import get_hmc_samples


def network(x_dim, scope, factor):
    with tf.variable_scope(scope):
        net = Sequential([
            Zip([
                Linear(x_dim, 10, scope='embed_1', factor=1.0 / 3),
                Linear(x_dim, 10, scope='embed_2', factor=factor * 1.0 / 3),
                Linear(2, 10, scope='embed_3', factor=1.0 / 3),
                lambda _: 0.,
            ]),
            sum,
            tf.nn.relu,
            Linear(10, 10, scope='linear_1'),
            tf.nn.relu,
            Parallel([
                Sequential([
                    Linear(10, x_dim, scope='linear_s', factor=0.001),
                    ScaleTanh(x_dim, scope='scale_s')
                ]),
                Linear(10, x_dim, scope='linear_t', factor=0.001),
                Sequential([
                    Linear(10, x_dim, scope='linear_f', factor=0.001),
                    ScaleTanh(x_dim, scope='scale_f'),
                ])
            ])
        ])

    return net

FOLDER = '/Users/Raza/VariationalSampling/data/german'

def load_data(folder):
    X = np.load(folder + '/data.npy')
    y = np.load(folder + '/labels.npy')
    return X, y

X, y = load_data(FOLDER)
x_dim = X.shape[1]
logisticregression = LogisticRegressionTF(X.astype(np.float32), y.astype(np.float32), data_dim=x_dim, prior_variance=0.1)


dynamics = Dynamics(x_dim, logisticregression.get_energy_func(), T=10, eps=0.1, net_factory=network, use_temperature=False)

x = tf.placeholder(tf.float32, shape=(None, x_dim))
z = tf.random_normal(tf.shape(x))

Lx, _, px, output = propose(x, dynamics, do_mh_step=True)
Lz, _, pz, _ = propose(z, dynamics, do_mh_step=False)

loss = 0.0

v1 = (tf.reduce_sum(tf.square(x - Lx), axis=1) * px) + 1e-4
v2 = (tf.reduce_sum(tf.square(z - Lz), axis=1) * pz) + 1e-4
scale = 1.0

loss += scale * (tf.reduce_mean(1.0 / v1) + tf.reduce_mean(1.0 / v2))
loss += (- tf.reduce_mean(v1) - tf.reduce_mean(v2)) / scale

global_step = tf.Variable(0., name='global_step', trainable=False)
learning_rate = tf.train.exponential_decay(1e-3, global_step, 1000, 0.96, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss, global_step=global_step)

import time

time1 = time.time()
n_steps = 100
n_samples = 200
losses = []

samples = np.random.randn(n_samples, x_dim)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter('logs/logistic_regression/', sess.graph)
train_writer.close()
temp = 0.5

for t in range(n_steps):
    print(t)
    _, loss_, samples, px_, lr_ = sess.run([
        train_op,
        loss,
        output[0],
        px,
        learning_rate,
    ], {x: samples, dynamics.temperature: temp})
    losses.append(loss_)
    temp = 0.5 + (t+1)/(2*n_steps)

    if t % 100 == 0:
        print(
            'Step: %d / %d, Loss: %.2e, Acceptance sample: %.2f, LR: %.5f' % (
            t, n_steps, loss_, np.mean(px_), lr_))
time2 = time.time()
print('Time to train sampler was {} seconds'.format(time2 - time1))

samples = distribution.get_samples(n=n_samples)
final_samples = []

for t in range(2000):
    final_samples.append(np.copy(samples))

    feed_dict = {
        x: samples, dynamics.temperature: 1.0,
    }

    samples = sess.run(output[0], feed_dict)

np.save('mogsamples_with_temp', np.array(final_samples))

L2HMC_samples = np.array(final_samples)
plt.plot(L2HMC_samples[:, 5, 0], L2HMC_samples[:, 5, 1], color='orange', marker='o', alpha=0.8)
plt.show()