from typing import List
import gym
import numpy as np
import tensorflow as tf
import tqdm
import config_file
import tensorboard


env = gym.make("CartPole-v1")
input_shape = 4 # == env.observation_space.shape
n_outputs = 2 # == env.action_space.n

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation="elu", input_shape=(input_shape, )),
    tf.keras.layers.Dense(32, activation="elu"),
    tf.keras.layers.Dense(n_outputs)
])


    
def epsilon_greedy_policy(state, epsilon=0.0):
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        Q_values = model.predict(state[np.newaxis], verbose=0) #type: ignore
        return np.argmax(Q_values[0])

from collections import deque
SIZE = 10000
action_history = deque(maxlen=SIZE)
state_history = deque(maxlen=SIZE)
state_next_history = deque(maxlen=SIZE)
rewards_history = deque(maxlen=SIZE)
done_history = deque(maxlen=SIZE)
episode_reward_history = deque(maxlen=SIZE)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

train_summary_writer = tf.summary.create_file_writer(config_file.LOG_DIR) #type: ignore

for episode in range(1000):
    state, _ = env.reset()

    episode_reward = 0

    for step in range(200):
        action = epsilon_greedy_policy(state, max(1-(episode/10), 0.01))

        state_next, reward, done, _, _ = env.step(action)
        state_next = np.array(state_next)

        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        rewards_history.append(reward)
        done_history.append(done)

        episode_reward += reward

        state = state_next

        if done:
            break

    idxs = np.random.randint(len(state_history), size=32)

    state_sample = np.array([state_history[i] for i in idxs])
    state_next_sample = np.array([state_next_history[i] for i in idxs])
    rewards_sample = np.array([rewards_history[i] for i in idxs])
    done_sample = np.array([float(done_history[i]) for i in idxs])
    actions_sample = np.array([action_history[i] for i in idxs])

    future_rewards = model.predict(state_next_sample, verbose=0) #type: ignore

    updated_q_values = rewards_sample + (1 - done_sample) * 0.95 * np.max(future_rewards, axis=1)

    masks = tf.one_hot(actions_sample, n_outputs)

    with tf.GradientTape() as tape:
        all_q_values = model(state_sample)
        q_values = tf.reduce_sum(all_q_values * masks, axis=1)

        loss = tf.reduce_mean(tf.square(updated_q_values - q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    with train_summary_writer.as_default():
        #tf.summary.scalar("loss", loss, step=episode)
        tf.summary.scalar("reward", episode_reward, step=episode)




        
