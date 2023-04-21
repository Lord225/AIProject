from typing import List, Tuple
import gym
import numpy as np
import tensorflow as tf
import tqdm
import config_file
import tensorboard
import random
from collections import deque

# RL implementation 4
# This implements the algorithm with the following changes:
# * uses advantege in calculating Q values
# * uses actor - critic loss
# * uses target network
# * uses replay buffer
# * uses double DQN

# env spec
env = gym.make("CartPole-v1")
input_shape = [4] # == env.observation_space.shape
n_outputs = 2 # == env.action_space.n

def get_network():
    inputs = tf.keras.layers.Input(shape=(4,))
    common = tf.keras.layers.Dense(32, 'elu')(inputs)
    common = tf.keras.layers.Dense(32, 'elu')(common)
    Q_values = tf.keras.layers.Dense(n_outputs)(common)
    critic = tf.keras.layers.Dense(1)(common)
    model = tf.keras.models.Model(inputs=[inputs], outputs=[Q_values, critic])

    return model

# networks
actor_model = get_network()
target_model = get_network()
target_model.set_weights(actor_model.get_weights())

RUN_VERSION = "v5.2"

train_summary_writer = tf.summary.create_file_writer(config_file.LOG_DIR+RUN_VERSION) #type: ignore

def env_step(action):
    state, reward, done, _, _ = env.step(action)
    return (state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action], [tf.float32, tf.float32, tf.int32]) #type: ignore

ReplayHistory = Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]

@tf.function
def run_episode(
        initial_state: tf.Tensor,
        actor_model: tf.keras.Model, 
        max_steps: int) -> ReplayHistory:
    """
    Run a single episode to collect training data
    collects: 
    * states - state at each step
    * actions - action taken at each step
    * rewards - reward received at each step
    * next_states - next state at each step
    * dones - done flag at each step
    """
    states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    next_states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    dones = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        states = states.write(t, state)

        state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        action_logits_t, _ = actor_model(state) # type: ignore

        # Sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1, dtype=tf.int32)[0, 0]
        actions = actions.write(t, action)

        # Apply action to the environment to get next state and reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        next_states = next_states.write(t, state)

        dones = dones.write(t, tf.cast(done, tf.float32))
        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    states = states.stack()
    actions = actions.stack()
    rewards = rewards.stack()
    next_states = next_states.stack()
    dones = dones.stack()

    return states, actions, rewards, next_states, dones

def get_expected_return(
    rewards: List, 
    gamma: float):
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    rewards = tf.cast(rewards[::-1], dtype=tf.float32) # type: ignore
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)

    returns = returns.stack()[::-1] # type: ignore

    return returns

BufferType = Tuple[tf.Tensor, tf.Tensor]

@tf.function
def run_episode_and_get_history(
        initial_state: tf.Tensor,
        actor_model: tf.keras.Model,
        max_steps: int,
        gamma: float
) -> Tuple[ReplayHistory, tf.Tensor]:
    # run whole episode
    states, action_probs, rewards, next_states, dones = run_episode(initial_state, actor_model, max_steps) # type: ignore

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma=gamma) #type: ignore

    return (states, action_probs, returns, next_states, dones), tf.reduce_sum(rewards)
import random


# optimized yey
# todo change this to a quicker version
def sample_experiences(batch_size, replay_memory):
    states, actions, rewards, next_states, dones = replay_memory
    # sample random experiences
    indices = random.sample(range(len(rewards)), batch_size)

    return (
        tf.gather(states, indices),
        tf.gather(actions, indices),
        tf.gather(rewards, indices),
        tf.gather(next_states, indices),
        tf.gather(dones, indices)
    )
    
batch_size = 1024
discount_rate = 0.99
episodes = 1000
train_iters_per_episode = 20
max_steps_per_episode = 200
target_update_freq = 50
minibatch_size = 128
replay_memory_size = 50000
save_freq = 250

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
loss_fn = tf.keras.losses.mean_squared_error

@tf.function
def training_step(
        batch: ReplayHistory,
        minibatch_size: int,
        train_iterations: int,
        target_model: tf.keras.Model,
        actor_model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer
):
    # sample minibatch_size experiences from batch
    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = batch

    for _ in tf.range(train_iterations):
        idxs = tf.random.uniform(shape=(minibatch_size,), minval=0, maxval=len(batch_states), dtype=tf.int32)

        states = tf.gather(batch_states, idxs)
        actions = tf.gather(batch_actions, idxs)
        rewards = tf.gather(batch_rewards, idxs)
        next_states = tf.gather(batch_next_states, idxs)
        dones = tf.gather(batch_dones, idxs)

        # next_Q_values, _ = target_model(next_states, training=True) # type: ignore
        # max_next_Q_values = tf.reduce_max(next_Q_values, axis=1)
        # target_Q_values = (rewards + (tf.constant(1.0, dtype=tf.float32) - dones) * discount_rate * max_next_Q_values)
        # target_Q_values = tf.reshape(target_Q_values, [-1, 1])
        # mask = tf.one_hot(actions, n_outputs)

        next_Q_values, _ = actor_model(next_states, training=True) # type: ignore
        best_next_actions = tf.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, n_outputs)
        next_best_Q, _ = target_model(next_states, training=True) # type: ignore 
        next_best_Q_values = tf.reduce_sum(next_best_Q * next_mask, axis=1)
        target_Q_values = (rewards + (tf.constant(1.0, dtype=tf.float32) - dones) * discount_rate * next_best_Q_values)
        target_Q_values = tf.reshape(target_Q_values, [-1, 1])
        mask = tf.one_hot(actions, n_outputs)

        with tf.GradientTape() as tape:
            all_Q_values, values = actor_model(states, training=True) # type: ignore
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            actor_loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
            critic_loss = tf.reduce_mean(loss_fn(target_Q_values, values))

            loss = actor_loss + critic_loss
        grads = tape.gradient(loss, actor_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, actor_model.trainable_variables))


    
def run():
    states_buffer = deque(maxlen=replay_memory_size)
    actions_buffer = deque(maxlen=replay_memory_size)
    returns_buffer = deque(maxlen=replay_memory_size)
    next_states_buffer = deque(maxlen=replay_memory_size)
    dones_buffer = deque(maxlen=replay_memory_size)

    t = tqdm.tqdm(range(episodes))
    for episode in t:
        # run episode
        state, _ = env.reset()
        state = tf.constant(state, dtype=tf.float32)
        (states, action_probs, returns, next_states, dones), total_rewards = run_episode_and_get_history(state, actor_model, max_steps_per_episode, discount_rate) #type: ignore

        # add to replay memory
        states_buffer.extend(states.numpy())
        actions_buffer.extend(action_probs.numpy())
        returns_buffer.extend(returns.numpy())
        next_states_buffer.extend(next_states.numpy())
        dones_buffer.extend(dones.numpy())

        # log
        with train_summary_writer.as_default():
            tf.summary.scalar('reward', total_rewards, step=episode)

        t.set_description(f"Episode {episode} - Reward: {total_rewards:.2f}")
        
        # train
        if len(states_buffer) > batch_size+1:
            batch = sample_experiences(batch_size, (states_buffer, actions_buffer, returns_buffer, next_states_buffer, dones_buffer))
    
            training_step(batch, minibatch_size, train_iters_per_episode, actor_model, actor_model, optimizer)
        # update target network
        if episode % target_update_freq == 0:
            target_model.set_weights(actor_model.get_weights())

        # save model
        if episode % save_freq == 0:
            actor_model.save(f"{config_file.MODELS_DIR}actor_model_{RUN_VERSION}_{config_file.RUN_NAME}_{episode}.h5")


run()