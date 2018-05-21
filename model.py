import math
import numpy as np
import random
import tensorflow as tf

update_sample_region = 1
num_update_samples = 1
num_update_iters = 1
input_size = 3
hidden_size = 35
learning_rate = 0.01
baseline_alpha = 0.1
train_iters = 100000
test_interval = 100
test_size = 100


def build_model(inputs):
    net = tf.layers.dense(inputs,
                          units=hidden_size,
                          activation=tf.nn.relu)
    net = tf.layers.dense(net,
                          units=hidden_size,
                          activation=tf.nn.relu)
    mean = tf.layers.dense(net, 1)
    mean = tf.squeeze(mean, -1)
    std = tf.layers.dense(net,
                          units=1,
                          activation=tf.nn.softplus)
    std = tf.squeeze(std, -1) * 10
    normal = tf.distributions.Normal(loc=mean, scale=std)
    return normal


def get_rewards(channel):
    if channel < 6:
        reward = channel
    else:
        reward = 12 - channel
    reward = reward / 10. + random.random()
    return reward


def get_update_samples(replay_memory, num_samples, baseline):
    total_num = len(replay_memory['channels'])
    num_samples = min(total_num, num_samples)
    channels = []
    rewards = []
    channels.append(replay_memory['channels'][total_num - 1])
    rewards.append(replay_memory['rewards'][total_num - 1])

    sample_region = min(update_sample_region, total_num)
    for _ in range(num_samples - 1):
        idx = int(sample_region * math.pow(random.random(), 1. / 8) + total_num - sample_region)
        channels.append(replay_memory['channels'][idx])
        rewards.append(replay_memory['rewards'][idx])

    rewards = [reward - baseline for reward in rewards]
    return channels, rewards


def deal_channel(channels):
    channel = int(channels[0])
    if channel > 10:
        channel = 10
    return channel

if __name__ == '__main__':
    data = tf.placeholder(tf.float32, shape=[None, input_size])
    channels = tf.placeholder(tf.float32, shape=[None])
    rewards = tf.placeholder(tf.float32, shape=[None])
    normal = build_model(data)
    sample = tf.squeeze(normal.sample(1), axis=0)
    n_log_p = - tf.log(normal.prob(channels + 1e-8))
    loss = tf.reduce_mean(n_log_p * rewards)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        replay_memory = {'channels': [], 'rewards': []}
        baseline = 0.
        beta_hat_t = 1.
        d = np.zeros([1, input_size], dtype=np.float32)
        for i in range(train_iters):
            c = sess.run(sample, feed_dict={data: d})
            c = deal_channel(c)
            r = get_rewards(c)
            replay_memory['channels'].append(c)
            replay_memory['rewards'].append(r)
            baseline = (1 - baseline_alpha) * baseline + baseline_alpha * r
            beta_hat_t *= 1 - baseline_alpha

            for _ in range(num_update_iters):
                c, r = get_update_samples(replay_memory=replay_memory,
                                          num_samples=num_update_samples,
                                          baseline=baseline / (1 - beta_hat_t))
                sess.run(train_op, feed_dict={
                    data: d,
                    channels: c,
                    rewards: r
                })

            if i % test_interval == 0:
                test_rewards = []
                for _ in range(test_size):
                    c = sess.run(sample, feed_dict={data: d})
                    c = deal_channel(c)
                    r = get_rewards(c)
                    test_rewards.append(r)
                average_reward = sum(test_rewards) / len(test_rewards)
                print('Iteration {}, baseline {}, average reward {}'.format(i, baseline, average_reward))
