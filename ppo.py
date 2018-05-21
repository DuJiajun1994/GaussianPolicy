import numpy as np
import random
import tensorflow as tf

input_size = 3
hidden_size = 35
learning_rate = 0.01
baseline_alpha = 0.1
train_iters = 100000
test_interval = 100
test_size = 100

ppo_epsilon = 0.2

def build_model(inputs, name, trainable):
    with tf.variable_scope(name):
        net = tf.layers.dense(inputs,
                              units=hidden_size,
                              activation=tf.nn.relu,
                              trainable=trainable)
        net = tf.layers.dense(net,
                              units=hidden_size,
                              activation=tf.nn.relu,
                              trainable=trainable)
        mean = tf.layers.dense(net, 1, trainable=trainable)
        mean = tf.squeeze(mean, -1)
        std = tf.layers.dense(net,
                              units=1,
                              activation=tf.nn.softplus,
                              trainable=trainable)
        std = tf.squeeze(std, -1) * 10
        normal = tf.distributions.Normal(loc=mean, scale=std)
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    return normal, variables


def get_rewards(channels):
    rewards = []
    for channel in channels:
        if channel < 6:
            reward = channel
        else:
            reward = 12 - channel
        reward = reward / 10. + random.random()
        rewards.append(reward)
    return rewards


def deal_channels(channels):
    for i in range(len(channels)):
        channel = int(channels[i])
        if channel > 10:
            channel = 10
        channels[i] = channel
    return channels

if __name__ == '__main__':
    data = tf.placeholder(tf.float32, shape=[None, input_size])
    channels = tf.placeholder(tf.float32, shape=[None])
    rewards = tf.placeholder(tf.float32, shape=[None])
    normal, variables = build_model(data, name='policy', trainable=True)
    old_normal, old_variables = build_model(data, name='old_policy', trainable=False)
    sample = tf.squeeze(normal.sample(1), axis=0)

    ratio = normal.prob(channels) / old_normal.prob(channels)
    objective = tf.minimum(ratio * rewards, tf.clip_by_value(ratio, 1. - ppo_epsilon, 1 + ppo_epsilon) * rewards)
    loss = - tf.reduce_mean(objective)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    update_old_policy_op = [old_variable.assign(variable) for variable, old_variable in zip(variables, old_variables)]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        baseline = 0.
        beta_hat_t = 1.
        d = np.zeros([1, input_size], dtype=np.float32)
        for i in range(train_iters):
            c = sess.run(sample, feed_dict={data: d})
            c = deal_channels(c)
            r = get_rewards(c)
            average_reward = sum(r) / len(r)
            baseline = (1 - baseline_alpha) * baseline + baseline_alpha * average_reward
            beta_hat_t *= 1 - baseline_alpha

            unbiased_baseline = baseline / (1 - beta_hat_t)
            r = [item - unbiased_baseline for item in r]
            sess.run(train_op, feed_dict={
                data: d,
                channels: c,
                rewards: r
            })

            sess.run(update_old_policy_op)

            if i % test_interval == 0:
                test_rewards = []
                for _ in range(test_size):
                    c = sess.run(sample, feed_dict={data: d})
                    deal_channels(c)
                    r = get_rewards(c)
                    test_rewards.append(sum(r) / len(r))
                average_reward = sum(test_rewards) / len(test_rewards)
                print('Iteration {}, baseline {}, average reward {}'.format(i, baseline, average_reward))
