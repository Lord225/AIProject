from typing import List
import gym
import numpy as np
import tensorflow as tf
import tqdm
import config_file

env = gym.make("CartPole-v1")
input_shape = [4] # == env.observation_space.shape
n_outputs = 2 # == env.action_space.n

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    tf.keras.layers.Dense(32, activation="elu"),
    tf.keras.layers.Dense(n_outputs)
])

from collections import deque
replay_memory_rewards = deque(maxlen=2000)
replay_memory_probs = deque(maxlen=2000)
replay_memory_states = deque(maxlen=2000)

def epsilon_greedy_policy(dist, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        return np.argmax(dist[0])

def run_episode(eps):
    state = env.reset()
    rewards = []
    probs = []
    states = []

    for step in range(1000):
        dist = model.predict(state[np.newaxis], verbose=0) # type: ignore
        dist = tf.nn.softmax(dist)

        action = epsilon_greedy_policy(dist, eps)
        log_prob = tf.math.log(dist[0][action])

        next_state, reward, done, _ = env.step(action)

        rewards.append(reward)
        probs.append(log_prob)
        states.append(state)
        
        if done:
            break
        state = next_state

    expected_rewards = calc_discounted_rewards(rewards, 0.99)

    replay_memory_rewards.extend(expected_rewards)
    replay_memory_probs.extend(probs)
    replay_memory_states.extend(states)

    return sum(rewards)

def calc_discounted_rewards(rewards, gamma):
    discounted_rewards = np.zeros(len(rewards))
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards


def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_memory_probs), size=batch_size)

    batch_probs = [replay_memory_probs[index] for index in indices]
    batch_rewards = [replay_memory_rewards[index] for index in indices]
    batch_states = [replay_memory_states[index] for index in indices]

    return batch_probs, batch_rewards, batch_states

optimizer = tf.keras.optimizers.Adam(lr=0.01)

def training_step(batch_size):
    _, batch_rewards, batch_states = sample_experiences(batch_size)

    with tf.GradientTape() as tape:
        Q_values = model(np.array(batch_states).reshape(-1, 4))
        loss = tf.reduce_mean(tf.math.log(Q_values) * batch_rewards)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    
rewards = [] 
best_score = 0

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_summary_writer = tf.summary.create_file_writer(config_file.LOG_DIR) #type: ignore
batch_size = 32

t = tqdm.tqdm(range(600))
for episode in t:
    epsilon = max(1 - episode / 500, 0.01)
    episode_reward = run_episode(epsilon)

    if episode > 50:
        training_step(batch_size)

    with train_summary_writer.as_default():
        tf.summary.scalar('reward', episode_reward, step=episode)
        tf.summary.scalar('loss', train_loss.result(), step=episode)
        train_loss.reset_states()
