import argparse

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from utils.func_utils import accept, jacobian, autocovariance, get_log_likelihood, \
    get_data, binarize, normal_kl, acl_spectrum, ESS
from utils.distributions import Gaussian, GMM, GaussianFunnel, gen_ring
from utils.layers import Linear, Sequential, Zip, Parallel, ScaleTanh
from utils.dynamics import Dynamics
from utils.sampler import propose
from utils.notebook_utils import  ensure_directory
from utils.evaluation import  batch_means_essTF


sess = tf.Session()

#%% set up distribution and constants

log_dir = 'logs/mog'
checkpoint_dir = 'checkpoints/mog/'
results_dir = 'results/'
x_dim = 2
n_steps = 8300
n_samples = 200
temp = 30
annealing_steps = 100
losses = []
means = [np.array([10., 0.0]).astype(np.float32), np.array([-10.0, 0.0]).astype(np.float32)]
covs = [np.array([[1.0, 0.0],[0.0, 1.0]]), np.array([[1.0, 0.0],[0.0, 1.0]])]
distribution = GMM(means, covs, [0.5, 0.5])
# Get some samples from the true distribution for debugging
init_samples = distribution.get_samples(200)
# plt.scatter(init_samples[:, 0], init_samples[:, 1])
# plt.show()
np.save('init_samples', init_samples)


#%%
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


#%%

dynamics = Dynamics(x_dim, distribution.get_energy_function(), T=10, eps=0.1, net_factory=network, use_temperature=True)

x = tf.placeholder(tf.float32, shape=(None, x_dim))
z = tf.random_normal(tf.shape(x))

Lx, _, px, output = propose(x, dynamics, do_mh_step=True)
Lz, _, pz, _ = propose(z, dynamics, do_mh_step=False)
# chain = tf.stack(output)
# ess = batch_means_essTF(chain)
# tf.summary.histogram('ess_y', ess[:, 0])
# tf.summary.histogram('ess_x', ess[:, 1])
# tf.summary.scalar('ess_min', tf.reduce_min(ess))
tf.summary.histogram('intermediate_sample_y', output[0][:, 1])
tf.summary.histogram('intermediate_sample_x', output[0][:, 0])

loss = 0.

v1 = (tf.reduce_sum(tf.square(x - Lx), axis=1) * px) + 1e-4
v2 = (tf.reduce_sum(tf.square(z - Lz), axis=1) * pz) + 1e-4
scale = 1

loss += scale * (tf.reduce_mean(1.0 / v1) + tf.reduce_mean(1.0 / v2))
loss += (- tf.reduce_mean(v1) - tf.reduce_mean(v2)) / scale
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0., name='global_step', trainable=False)
learning_rate = tf.train.exponential_decay(1e-3, global_step, 1000, 0.96, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss, global_step=global_step)

#%%
ensure_directory(log_dir)
ensure_directory(checkpoint_dir)
summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())


import time
training_time = 0.0
samples = np.random.randn(n_samples, x_dim)

intermediate_samples = []

for t in range(n_steps):
    time1 = time.time()
    _, loss_, samples, px_, lr_ = sess.run([
        train_op,
        loss,
        output[0],
        px,
        learning_rate,
    ], {x: samples, dynamics.temperature: temp})
    time2 = time.time()
    training_time += time2 - time1
    losses.append(loss_)
    intermediate_samples.append(samples)
    if t % annealing_steps == 0:
        print(
            'Time: %d, Step: %d / %d, Loss: %.2e, Acceptance sample: %.2f, LR: %.5f, Temp: %.5f' % (
            training_time, t, n_steps, loss_, np.mean(px_), lr_, temp))
        sumstr =sess.run(summary, {x: samples, dynamics.temperature: temp})
        summary_writer.add_summary(sumstr)
        # temp_samples = distribution.get_samples(200, T=temp)
        # plt.scatter(temp_samples[:, 0], temp_samples[:, 1])
        # plt.show()
        if temp > 1.0:
            temp *= 0.96

    if (t + 1) % 2000:
        saver.save(sess, checkpoint_dir + 'mog_sampler', global_step)

np.save('intermediate_samples_', np.array(intermediate_samples))


print('Time to train sampler was {} seconds'.format(training_time))

#%%
final_samples = []
samples = np.random.normal(size=(n_samples, 2))

time1 = time.time()
for t in range(20000):
    feed_dict = {
        x: samples, dynamics.temperature: 1.0,
    }
    samples = sess.run(output[0], feed_dict)
    final_samples.append(np.copy(samples))
time2 = time.time()
sample_time = (time2 - time1)/(10000 * n_samples)

ensure_directory(results_dir)
np.savez(results_dir + 'mogsamples_final', np.array(final_samples), training_time, sample_time)


