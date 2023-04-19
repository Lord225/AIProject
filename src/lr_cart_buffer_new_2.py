from typing import List, Tuple
import gym
import numpy as np
import tensorflow as tf
import tqdm
import config_file
import tensorboard

# shit
env = gym.make("CartPole-v1")
input_shape = [4] # == env.observation_space.shape
n_outputs = 2 # == env.action_space.n

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    tf.keras.layers.Dense(32, activation="elu"),
    tf.keras.layers.Dense(n_outputs)
])

train_summary_writer = tf.summary.create_file_writer(config_file.LOG_DIR) #type: ignore

def env_step(action):
    state, reward, done, _, _ = env.step(action)
    return (state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action], [tf.float32, tf.float32, tf.int32]) #type: ignore

ReplayHistory = Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]

@tf.function
def run_episode(
        initial_state: tf.Tensor,
        model: tf.keras.Model, 
        max_steps: int) -> ReplayHistory:
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
        action_logits_t = model(state)

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

    return returns

BufferType = Tuple[tf.Tensor, tf.Tensor]

@tf.function
def run_episode_and_get_history(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        max_steps: int,
        gamma: float
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    states, action_probs, rewards, next_states, dones = run_episode(initial_state, model, max_steps)

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma=gamma) #type: ignore

    return states, action_probs, returns, next_states, dones, tf.reduce_sum(rewards)

# todo change this to a quicker version
def sample_experiences(batch_size, replay_memory) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)
    ]


    return states, actions, rewards, next_states, dones

batch_size = 32
discount_rate = 0.95
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.mean_squared_error

# def training_step(batch_size):
#     experiences = sample_experiences(batch_size)
#     states, actions, rewards, next_states, dones = experiences
#     next_Q_values = model.predict(next_states, verbose=0)
#     max_next_Q_values = np.max(next_Q_values, axis=1)
#     target_Q_values = (rewards +(1 - dones) * discount_rate * max_next_Q_values)
#     target_Q_values = target_Q_values.reshape(-1, 1)
#     mask = tf.one_hot(actions, n_outputs)
#     with tf.GradientTape() as tape:
#         all_Q_values = model(states)
#         Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
#         loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
@tf.function
def training_step(
        batch: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer
):
    # states, actions, rewards, next_states, dones = batch
    # next_Q_values = model.predict(next_states, verbose=0)
    # max_next_Q_values = np.max(next_Q_values, axis=1)
    # target_Q_values = (rewards + (tf.constant(1.0, dtype=tf.float32) - dones) * discount_rate * max_next_Q_values)
    # target_Q_values = target_Q_values.reshape(-1, 1)
    # mask = tf.one_hot(actions, n_outputs)
    # with tf.GradientTape() as tape:
    #     all_Q_values = model(states)
    #     Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
    #     loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    # grads = tape.gradient(loss, model.trainable_variables)
    # optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # tf function version
    states, actions, rewards, next_states, dones = batch
    next_Q_values = model(next_states, training=True)
    max_next_Q_values = tf.reduce_max(next_Q_values, axis=1)
    target_Q_values = (rewards + (tf.constant(1.0, dtype=tf.float32) - dones) * discount_rate * max_next_Q_values)
    target_Q_values = tf.reshape(target_Q_values, [-1, 1])
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states, training=True)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


from collections import deque
    
def run():
    buffer = deque(maxlen=10000)
    t = tqdm.tqdm(range(1000))
    for episode in t:
        state, _ = env.reset()
        state = tf.constant(state, dtype=tf.float32)
        states, action_probs, returns, next_states, dones, total_rewards = run_episode_and_get_history(state, model, 200, 0.99) #type: ignore

        for _state, _action_probs, _returns, _next_state, _done in zip(states, action_probs, returns, next_states, dones):
            buffer.append((_state, _action_probs, _returns, _next_state, _done))

        with train_summary_writer.as_default():
            tf.summary.scalar('reward', total_rewards, step=episode)
        
        if len(buffer) > 100:
            for i in range(10):
                batch = sample_experiences(32, buffer)
        
                training_step(batch, model, optimizer)


run()