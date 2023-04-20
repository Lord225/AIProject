from typing import List
import gym
import numpy as np
import tensorflow as tf
import tqdm
import config_file

import tensorboard
# works fine, add critic loss

env = gym.make("CartPole-v1", render_mode="human")
input_shape = [4] # == env.observation_space.shape
n_outputs = 2 # == env.action_space.n

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    tf.keras.layers.Dense(32, activation="elu"),
    tf.keras.layers.Dense(n_outputs)
])

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        Q_values = model.predict(state[np.newaxis], verbose=0) #type: ignore
        return np.argmax(Q_values[0])
    

from collections import deque

replay_memory = deque(maxlen=10000)

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, *info = env.step(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info


batch_size = 128
batch_size = 128
discount_rate = 0.95
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
loss_fn = tf.keras.losses.mean_squared_error
# 2279
# * jira & temp
# * Test sprawidź jak działa
# * Bd sprawidź
# 
def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states, verbose=0) #type: ignore
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards + (1 - dones) * discount_rate * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

rewards = [] 
best_score = 0

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_summary_writer = tf.summary.create_file_writer(config_file.LOG_DIR) #type: ignore

t = tqdm.tqdm(range(600), desc="Episode", unit="episode")

for episode in t:
    obs, *_ = env.reset()    

    env.render()

    episode_reward = 0
    for step in range(200):
        epsilon = max(1 - episode / 50, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        episode_reward += reward
        if done:
            break

    if episode > 10:
        for i in range(10):
            training_step(batch_size)

    with train_summary_writer.as_default():
        tf.summary.scalar('reward', episode_reward, step=episode)
        tf.summary.scalar('loss', train_loss.result(), step=episode)
        train_loss.reset_states()
