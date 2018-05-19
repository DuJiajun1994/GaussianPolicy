import math
import numpy as np
import random
import tensorflow as tf


replay_region = 100
num_replay_samples = 32
num_update_iters = 8
input_size = 3
hidden_size = 35
learning_rate = 0.01
baseline_alpha = 0.1
train_iters = 100000
test_interval = 100
test_size = 100


class Net:
    def __init__(self, channel, reward):
        self.channel = channel
        self.reward = reward


def build_model(inputs):
    net = tf.layers.dense(inputs,
                          units=hidden_size,
                          activation=tf.nn.relu)
    net = tf.layers.dense(net,
                          units=hidden_size,
                          activation=tf.nn.relu)
    mean = tf.layers.dense(net, 1)
    mean = tf.squeeze(mean, -1)
    log_std = tf.layers.dense(net, 1)
    log_std = tf.clip_by_value(log_std, -3, 1)
    log_std = tf.squeeze(log_std, -1)
    return mean, log_std


def get_rewards(channels):
    batch_size = channels.shape[0]
    rewards = np.empty(shape=[batch_size], dtype=np.float32)
    for i in range(batch_size):
        channel = channels[i]
        if channel < 6:
            reward = channel
        else:
            reward = 12 - channel
        rewards[i] = reward / 10. + random.random()
    return rewards


def sample_channels(mean, log_std):
    batch_size = len(mean)
    channels = np.empty([batch_size], dtype=np.float32)
    for i in range(batch_size):
        std = math.exp(log_std[i])
        channel = random.gauss(mean[i], std)
        if channel > 10:
            channel = 10
        channels[i] = channel
    return channels


def get_replay_samples(replay_memory):
    total_num = min(len(replay_memory), replay_region)
    num_samples = min(total_num, num_replay_samples)
    channels = []
    rewards = []
    for _ in range(num_samples):
        idx = random.randint(0, total_num - 1)
        net = replay_memory[idx]
        assert isinstance(net, Net)
        channels.append(net.channel)
        rewards.append(net.reward)
    mean_reward = sum(rewards) / len(rewards)
    rewards = [reward - mean_reward for reward in rewards]
    return channels, rewards

if __name__ == '__main__':
    data = tf.placeholder(tf.float32, shape=[None, input_size])
    channels = tf.placeholder(tf.float32, shape=[None])
    rewards = tf.placeholder(tf.float32, shape=[None])
    mean, log_std = build_model(data)
    n_log_p = log_std + tf.square(channels - mean) / (tf.exp(log_std * 2) * 2)
    losses = n_log_p * rewards
    loss = tf.reduce_mean(losses)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        replay_memory = []
        baseline = 0.
        beta_hat_t = 1.
        d = np.zeros([1, input_size], dtype=np.float32)
        for i in range(train_iters):
            m, l = sess.run([mean, log_std], feed_dict={data: d})
            c = sample_channels(m, l)
            r = get_rewards(c)
            average_reward = sum(r) / len(r)
            baseline = (1 - baseline_alpha) * baseline + baseline_alpha * average_reward
            beta_hat_t *= 1 - baseline_alpha

            replay_memory.append(Net(channel=c, reward=r))
            replay_memory.sort(key=lambda a: a.reward, reverse=True)

            for _ in range(num_update_iters):
                unbiased_baseline = baseline / (1 - beta_hat_t)
                update_channels, update_rewards = get_replay_samples(replay_memory)
                update_channels.append(c)
                update_rewards.append(r - unbiased_baseline)
                sess.run(train_op, feed_dict={
                    data: d,
                    channels: c,
                    rewards: r
                })

            if i % test_interval == 0:
                test_rewards = []
                for _ in range(test_size):
                    m, l = sess.run([mean, log_std], feed_dict={data: d})
                    c = sample_channels(m, l)
                    r = get_rewards(c)
                    test_rewards.append(sum(r) / len(r))
                average_reward = sum(test_rewards) / len(test_rewards)
                print('Iteration {}, baseline {}, average reward {}'.format(i, baseline, average_reward))
