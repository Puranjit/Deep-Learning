# We'll start with our library imports...
from __future__ import print_function
import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
import gym                         # to setup and run RL environments
import scipy.signal                # for a specific convolution function

env = gym.make('CartPole-v1')
NUM_ACTIONS = env.action_space.n
OBS_SHAPE = env.observation_space.shape

network_fn = lambda shape: tf.keras.Sequential([
                            tf.keras.layers.Dense(256, activation=tf.nn.tanh),
                            tf.keras.layers.Dense(256, activation=tf.nn.tanh),
                            tf.keras.layers.Dense(shape)])

policy_network = network_fn(NUM_ACTIONS)
value_network = network_fn(1)
state = env.reset()
input_ = np.expand_dims(state, 0)

print("Policy action logits:", policy_network(input_))
print("Estimated Value:", value_network(input_))

VALUE_FN_ITERS = 80
POLICY_FN_ITERS = 80
KL_MARGIN = 1.2
KL_TARGET = 0.01
CLIP_RATIO = 0.2
optimizer = tf.keras.optimizers.Adam()

def discount_cumsum(x, discount):
    """
    magic from the rllab library for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1],
                                axis=0)[::-1]
def categorical_kl(logp0, logp1):
    """Returns average kl divergence between two batches of distributions"""
    all_kls = tf.reduce_sum(tf.exp(logp1) * (logp1 - logp0), axis=1)
    return tf.reduce_mean(all_kls)

def update_fn(policy_network, value_network, states, actions, rewards, gamma=0.99):

    vals = np.squeeze(value_network(states))
    deltas = rewards[:-1] + gamma * vals[1:] - vals[:-1]
    advantage = discount_cumsum(deltas, gamma)

    action_logits = policy_network(states)
    initial_all_logp = tf.nn.log_softmax(action_logits)
    row_indices = tf.range(initial_all_logp.shape[0])
    indices = tf.transpose([row_indices, actions])
    initial_action_logp = tf.gather_nd(initial_all_logp, indices)
    
    for _ in range(POLICY_FN_ITERS):
        with tf.GradientTape() as tape:
            # get the policy's action probabilities
            action_logits = policy_network(states)
            all_logp = tf.nn.log_softmax(action_logits)
            row_indices = tf.range(all_logp.shape[0])
            indices = tf.transpose([row_indices, actions])
            action_logp = tf.gather_nd(all_logp, indices)
            # decide how much to reinforce
            ratio = tf.exp(action_logp - tf.stop_gradient(initial_action_logp))
            min_adv = tf.where(advantage > 0.,
                               tf.cast((1.+CLIP_RATIO)*advantage, tf.float32), 
                               tf.cast((1.-CLIP_RATIO)*advantage, tf.float32)
                               )
            surr_adv = tf.reduce_mean(tf.minimum(ratio[:-1] * advantage, min_adv))
            pi_objective = surr_adv
            pi_loss = -1. * pi_objective

        grads = tape.gradient(pi_loss, policy_network.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_network.trainable_variables))
        
        kl = categorical_kl(all_logp, initial_all_logp)
        if kl > KL_MARGIN * KL_TARGET:
            break
            
    returns = discount_cumsum(rewards, gamma)[:-1]
    
    for _ in range(VALUE_FN_ITERS):
        with tf.GradientTape() as tape:
            vals = value_network(states)[:-1]
            val_loss = (vals - returns)**2

        grads = tape.gradient(val_loss, value_network.trainable_variables)
        optimizer.apply_gradients(zip(grads, value_network.trainable_variables))

state_buffer = []
action_buffer = []
reward_buffer = []
done = False
state = env.reset()
i = 0

while not done:
    state_buffer.append(state)
    action_logits = policy_network(np.expand_dims(state, 0))
    action = np.squeeze(tf.random.categorical(action_logits, 1))
    action_buffer.append(action)
    state, rew, done, info = env.step(action)
    reward_buffer.append(rew)

states = np.stack(state_buffer)
actions = np.array(action_buffer)
rewards = np.array(reward_buffer)
update_fn(policy_network, value_network, states, actions, rewards)

def categorical_entropy(logits):
    return -1 * tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(logits) * tf.nn.log_softmax(logits), axis=-1))

count = 0
length = []
number_of_eps = []
loss = []

for episode in range(25):
    
    done = False
    state = env.reset()
    y = 0
    count = 0
    while not done:
        state_buffer.append(state)
        
        count += 1
        
        action_logits = policy_network(np.expand_dims(state, 0))
        action = np.squeeze(tf.random.categorical(action_logits, 1))
        action_buffer.append(action)

        state, rew, done, info = env.step(action)
        reward_buffer.append(rew)
        
        states = np.stack(state_buffer)
        actions = np.array(action_buffer)
        rewards = np.array(reward_buffer)
        update_fn(policy_network, value_network, states, actions, rewards)
        
    y = categorical_entropy(action_logits)
    loss.append(y)
    
    print("episode:", episode," Length:", count)
    
    number_of_eps.append(int(episode))
    length.append(int(count))

plt.scatter(length, loss)
plt.xlabel('Episode length')
plt.ylabel('Mean entropy')
plt.show()

plt.plot(number_of_eps, loss, 'r')
plt.xlabel('Number of Episodes')
plt.ylabel('Mean entropy')
plt.show()