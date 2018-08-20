import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils.func_utils import accept, jacobian, autocovariance, get_log_likelihood, \
    get_data, binarize, normal_kl, acl_spectrum, ESS
from utils.distributions import get_donut_energy
from utils.layers import Linear, Sequential, Zip, Parallel, ScaleTanh
from utils.dynamics import Dynamics
from utils.sampler import propose
from utils.notebook_utils import ensure_directory

sess = tf.Session()

#%% Set constants
log_dir = 'logs/ring_experiments/'
results_dir = 'results/ring_experiments/'
checkpoints = 'checkpoints/ring/'
ensure_directory(log_dir)
ensure_directory(results_dir)
ensure_directory(checkpoints)
x_dim = 2
n_steps = 5000
n_samples = 200

#%% Setup network
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

#%% Set the distributions

dynamics = Dynamics(x_dim, get_donut_energy(), T=10, eps=0.1, net_factory=network, use_temperature=False)

x = tf.placeholder(tf.float32, shape=(None, x_dim))
z = tf.random_normal(tf.shape(x))

Lx, _, px, output = propose(x, dynamics, do_mh_step=True)
Lz, _, pz, _ = propose(z, dynamics, do_mh_step=False)

loss = 0.0

v1 = (tf.reduce_sum(tf.square(x - Lx), axis=1) * px) + 1e-4
v2 = (tf.reduce_sum(tf.square(z - Lz), axis=1) * pz) + 1e-4
scale = 0.1

loss += scale * (tf.reduce_mean(1.0 / v1) + tf.reduce_mean(1.0 / v2))
loss += (- tf.reduce_mean(v1) - tf.reduce_mean(v2)) / scale

global_step = tf.Variable(0.0, name='global_step', trainable=False)
learning_rate = tf.train.exponential_decay(1e-3, global_step, 1000, 0.96, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss, global_step=global_step)

losses = []

samples = np.random.randn(n_samples, x_dim)

sess.run(tf.global_variables_initializer())

#%% Train sampler
train_time = 0.0
for t in range(n_steps):
    time1 = time.time()
    _, loss_, samples, px_, lr_ = sess.run([
        train_op,
        loss,
        output[0],
        px,
        learning_rate,
    ], {x: samples})
    time2 = time.time()
    train_time += time2 - time1
    losses.append(loss_)

    if t % 100 == 0:
        print(
            'Step: %d / %d, Loss: %.2e, Acceptance sample: %.2f, LR: %.5f' % (
            t, n_steps, loss_, np.mean(px_), lr_))


print('Time to train sampler was {} seconds'.format(train_time))


#%% Draw some samples
final_samples = []

sample_time = 0.0
for t in range(1000):

    final_samples.append(np.copy(samples))

    feed_dict = {
        x: samples,
    }
    time1 = time.time()
    samples = sess.run(output[0], feed_dict)
    time2 = time.time()
    sample_time += time2 - time1

np.savez('ringsamples', np.array(final_samples), train_time, sample_time)
