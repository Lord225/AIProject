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


def epsilon_greedy_policy(state: tf.Tensor, epsilon=0.0):
    if tf.random.uniform(shape=(), minval=0, maxval=1) < epsilon:
        return tf.random.uniform(shape=(), minval=0, maxval=n_outputs, dtype=tf.int32)
    else:
        Q_values = model(tf.expand_dims(state, axis=0), training=True)
        return tf.argmax(Q_values[0], dtype=tf.int32) #type: ignore
    

ReplayBufferType = Tuple[tf.TensorArray, tf.TensorArray, tf.TensorArray, tf.TensorArray, tf.TensorArray]
BatchType = Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]

def sample_experiences(batch_size: int, 
                       replay_buffer: ReplayBufferType) -> BatchType:
    states, actions, rewards, next_states, dones = replay_buffer
    indices = tf.random.uniform(shape=(batch_size,), minval=0, maxval=states.size(), dtype=tf.int32)

    batch_states = states.gather(indices)
    batch_actions = actions.gather(indices)
    batch_rewards = rewards.gather(indices)
    batch_next_states = next_states.gather(indices)
    batch_dones = dones.gather(indices)

    return (batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)


def run_episode(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        replay_buffer: ReplayBufferType, 
        max_steps: int,
        replay_buffer_size: int) -> Tuple[ReplayBufferType, tf.Variable]: 
    # replay buffer is states, actions, rewards, next_states, dones
    states, actions, rewards, next_states, dones = replay_buffer
    
    # get size of TensorArray replay buffer
    size = states.size()

    initial_state_shape = initial_state.shape
    state = initial_state

    rewards_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)


    for t in tf.range(max_steps):
        index = (size + t) % replay_buffer_size # błąd

        # Convert state into a batched tensor (batch size = 1)
        state_batched = tf.expand_dims(state, 0)

        # Append state to the states tensor array
        states = states.write(index, state)

        # Run the model and to get action probabilities and critic value
        action_logits_t = epsilon_greedy_policy(state_batched, 0.1)

        # Sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1, dtype=tf.int32)[0, 0]
        #action_probs_t = tf.nn.softmax(action_logits_t)

        # Store log probability of the action chosen
        actions = actions.write(index, action) # type: ignore

        # Apply action to the environment to get next state and reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        # add next_states to replay buffer
        next_states = next_states.write(size + t, state)


        # Store reward
        rewards = rewards.write(index, tf.cast(reward, tf.float32))
        rewards_array = rewards_array.write(t, tf.cast(reward, tf.float32))

        # Store done
        dones = dones.write(index, tf.cast(done, tf.int32))

        if tf.cast(done, tf.bool):
            break

    return (states, actions, rewards, next_states, dones), tf.reduce_sum(rewards_array.stack())

batch_size = 32
discount_rate = 0.95
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
loss_fn = tf.keras.losses.mean_squared_error

# def training_step(batch_size):
#     experiences = sample_experiences(batch_size)
#     states, actions, rewards, next_states, dones = experiences
#     next_Q_values = model.predict(next_states, verbose=0)
#     max_next_Q_values = np.max(next_Q_values, axis=1)
#     target_Q_values = (rewards + (1 - dones) * discount_rate * max_next_Q_values)
#     target_Q_values = target_Q_values.reshape(-1, 1)
#     mask = tf.one_hot(actions, n_outputs)
#     with tf.GradientTape() as tape:
#         all_Q_values = model(states)
#         Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
#         loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))




        


# rewards = [] 
# best_score = 0

# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_summary_writer = tf.summary.create_file_writer(config_file.LOG_DIR) #type: ignore


# for episode in range(600):
#     obs, *_ = env.reset()    
#     episode_reward = 0
#     for step in range(200):
#         epsilon = max(1 - episode / 500, 0.01)
#         obs, reward, done, info = play_one_step(env, obs, epsilon)
#         episode_reward += reward
#         if done:
#             break

#     if episode > 50:
#         training_step(batch_size)

#     with train_summary_writer.as_default():
#         tf.summary.scalar('reward', episode_reward, step=episode)
#         tf.summary.scalar('loss', train_loss.result(), step=episode)
#         train_loss.reset_states()


replay_buffer_size = 1000
episodes = 10000
max_steps = 200

def reset_env():
    initial_state, _ = env.reset()
    #initial_state = tf.constant(initial_state, dtype=tf.float32)
    return initial_state.astype(np.float32)
    

@tf.function
def run():
    replay_buffer = (tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True), # states
                    tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True),    # actions
                    tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),  # rewards
                    tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),  # next_states
                    tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True))    # dones
    
    for episode in tf.range(episodes):
        initial_state = tf.numpy_function(reset_env, [], tf.float32)

        with tf.GradientTape() as tape:
            replay_buffer, reward_sum = run_episode(initial_state,
                        model,
                        replay_buffer,
                        max_steps,
                        replay_buffer_size)
            
            # get replay buffer size
            sample = sample_experiences(batch_size, replay_buffer)

            # train model   
            states, actions, rewards, next_states, dones = sample

            next_Q_values = model(next_states)
            max_next_Q_values = tf.reduce_max(next_Q_values, axis=1)
            target_Q_values = (rewards + (tf.constant(1, dtype=tf.float32) - tf.cast(dones, tf.float32)) * discount_rate * max_next_Q_values)
            target_Q_values = tf.expand_dims(target_Q_values, axis=1)
            mask = tf.one_hot(actions, n_outputs)

            all_Q_values = model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=tf.cast(episode, dtype=tf.int64))
                tf.summary.scalar('reward', reward_sum, step=tf.cast(episode, dtype=tf.int64))



    return replay_buffer[0].stack()
     
print(run())