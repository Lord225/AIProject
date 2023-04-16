# import tensorboard
# import tensorflow as tf
# import config_file
# from collections import deque
# import numpy as np
# import gym


# def make_model(n_outputs):
#     model = tf.keras.Sequential()

#     model.add(tf.keras.layers.Input(shape=(4,)))
#     model.add(tf.keras.layers.Dense(32, activation='elu'))
#     model.add(tf.keras.layers.Dense(32, activation='elu'))
#     model.add(tf.keras.layers.Dense(n_outputs))

#     return model

# env = gym.make('CartPole-v0')
# input_shape = [4]
# n_outputs = 2

# model = make_model(n_outputs)

# def greedy_policy(state, eps=0):
#     if np.random.rand() < eps:
#         return np.random.randint(2)
#     else:
#         Q_values = model.predict(state[np.newaxis])
#         return np.argmax(Q_values)

# buffer = deque(maxlen=2000)

# def sample_experience(batch_size):
#     idxs = np.random.randint(len(buffer), size=batch_size)

#     batch = [buffer[idx] for idx in idxs]

#     states, actions, rewards, next_states, dones = [
#         np.array([experience[field_idx] for experience in batch])
#         for field_idx in range(5)
#     ]

#     return states, actions, rewards, next_states, dones

# def play_one_step(env, state, epsilon):
#     action = greedy_policy(state, epsilon)
#     result = env.step(action)
#     next_state, reward, done, x, info = result
#     buffer.append((state, action, reward, next_state, done))

#     return next_state, reward, done, info


# optimizer = tf.keras.optimizers.Adam(lr=0.001)
# loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

# train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)


# def train_step(batch_size):
#     exp = sample_experience(batch_size)

#     states, actions, rewards, next_states, done = exp

#     next_Q_values = model.predict(next_states, verbose=0)
#     max_next_Q_values = np.max(next_Q_values, axis=1)

#     target_Q_values = (rewards + (1 - done) * 0.95 * max_next_Q_values)

#     mask = tf.one_hot(actions, n_outputs)

#     with tf.GradientTape() as tape:
#         all_Q_values = model(states)

#         Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)

#         loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

#         train_loss(loss)

#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))


# # create tensorboard to monitor training
# train_summary_writer = tf.summary.create_file_writer(config_file.LOG_DIR)


# for episode in range(2000):
#     state, info = env.reset()

#     reward_sums = 0
#     for step in range(200):
#         epsilon = max(1 - episode / 500, 0.01)
#         state, reward, done, info = play_one_step(env, state, epsilon)

#         reward_sums += reward

#         if done:
#             break

#     with train_summary_writer.as_default():
#         tf.summary.scalar('loss', train_loss.result(), step=episode)
#         tf.summary.scalar('rewards', reward_sums, step=episode)

#         train_loss.reset_states()

#     if episode > 50:
#         train_step(128)
    

# works well & have critic but do not have buffer  

import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple


# Create the environment
env = gym.make("CartPole-v1")

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self, 
      num_actions: int, 
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.common = layers.Dense(num_hidden_units, activation="relu")
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.common(inputs)
    return self.actor(x), self.critic(x)
  

num_actions =  2
num_hidden_units = 128

model = ActorCritic(num_actions, num_hidden_units)

# Wrap Gym's `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""

  state, reward, done, truncated, info = env.step(action)
  return (state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action], [tf.float32, tf.int32, tf.int32])



def run_episode(
    initial_state: tf.Tensor,  
    model: tf.keras.Model, 
    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Runs a single episode to collect training data."""

  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state

  for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    state = tf.expand_dims(state, 0)

    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)

    # Sample next action from the action probability distribution
    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    action_probs_t = tf.nn.softmax(action_logits_t)

    # Store critic values
    values = values.write(t, tf.squeeze(value))

    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0, action])

    # Apply action to the environment to get next state and reward
    state, reward, done = tf_env_step(action)
    state.set_shape(initial_state_shape)

    # Store reward
    rewards = rewards.write(t, reward)

    if tf.cast(done, tf.bool):
      break

  action_probs = action_probs.stack()
  values = values.stack()
  rewards = rewards.stack()

  return action_probs, values, rewards


def get_expected_return(
    rewards: tf.Tensor, 
    gamma: float, 
    standardize: bool = True) -> tf.Tensor:
  """Compute expected returns per timestep."""

  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)

  # Start from the end of `rewards` and accumulate reward sums
  # into the `returns` array
  rewards = tf.cast(rewards[::-1], dtype=tf.float32)
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + gamma * discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
  returns = returns.stack()[::-1]

  if standardize:
    returns = ((returns - tf.math.reduce_mean(returns)) / 
               (tf.math.reduce_std(returns) + eps))

  return returns


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(
    action_probs: tf.Tensor,  
    values: tf.Tensor,  
    returns: tf.Tensor) -> tf.Tensor:
  """Computes the combined Actor-Critic loss."""

  advantage = returns - values

  action_log_probs = tf.math.log(action_probs)
  actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

  critic_loss = huber_loss(values, returns)

  return actor_loss + critic_loss


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


@tf.function
def train_step(
    initial_state: tf.Tensor, 
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer, 
    gamma: float, 
    max_steps_per_episode: int) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:

    # Run the model for one episode to collect training data
    action_probs, values, rewards = run_episode(
        initial_state, model, max_steps_per_episode) 

    # Calculate the expected returns
    returns = get_expected_return(rewards, gamma)

    # Convert training data to appropriate TF tensor shapes
    action_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

    # Calculate the loss values to update our network
    loss = compute_loss(action_probs, values, returns)

  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)

  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward

min_episodes_criterion = 100
max_episodes = 10000
max_steps_per_episode = 500

# `CartPole-v1` is considered solved if average reward is >= 475 over 500 
# consecutive trials
reward_threshold = 475
running_reward = 0

# The discount factor for future rewards
gamma = 0.99

# Keep the last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

t = tqdm.trange(max_episodes)
for i in t:
    initial_state, info = env.reset()
    initial_state = tf.constant(initial_state, dtype=tf.float32)
    episode_reward = int(train_step(
        initial_state, model, optimizer, gamma, max_steps_per_episode))

    episodes_reward.append(episode_reward)
    running_reward = statistics.mean(episodes_reward)


    t.set_postfix(
        episode_reward=episode_reward, running_reward=running_reward)

    # Show the average episode reward every 10 episodes
    if i % 10 == 0:
      pass # print(f'Episode {i}: average reward: {avg_reward}')

    if running_reward > reward_threshold and i >= min_episodes_criterion:  
        break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')