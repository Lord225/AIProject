from typing import List, Tuple
import gym
import numpy as np
import tensorflow as tf
import tqdm
import config_file

# works fine but do not have buffer

# shit

# env = gym.make("CartPole-v1")

# inputs = tf.keras.layers.Input(shape=(4,))
# x = tf.keras.layers.Dense(32, activation="elu")(inputs)
# x = tf.keras.layers.Dense(32, activation="elu")(x)
# actor = tf.keras.layers.Dense(2)(x)
# critic = tf.keras.layers.Dense(1)(x)

# model = tf.keras.Model(inputs=inputs, outputs=[actor, critic])

# def env_step(action):
#     state, reward, done, _, _ = env.step(action)
#     return (state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32))

# def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
#   return tf.numpy_function(env_step, [action], [tf.float32, tf.float32, tf.int32]) # type: ignore

# def env_reset():
#    state, _ = env.reset()
#    return state.astype(np.float32)

# def tf_env_reset() -> tf.Tensor:
#     return tf.numpy_function(env_reset, [], [tf.float32]) # type: ignore

# HistoryType = Tuple[tf.TensorArray, tf.TensorArray, tf.TensorArray, tf.TensorArray]

# def run_episode(
#     initial_state: tf.Tensor,  
#     model: tf.keras.Model, 
#     max_steps: int,
#     history: HistoryType,
#     history_index: int,
#     history_size: int,
#     ) -> Tuple[HistoryType, int]:
#     """Runs a single episode to collect training data."""
    
#     states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
#     action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
#     values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
#     rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    
#     initial_state_shape = initial_state
#     state = initial_state

#     num_episodes = 0
#     for t in tf.range(max_steps):
#         # Convert state into a batched tensor (batch size = 1)
#         states.write(t, state)
#         state = tf.expand_dims(state, 0)

#         # Run the model and to get action probabilities and critic value
#         action_logits_t, value = model(state)
#         num_episodes += 1

#         # Sample next action from the action probability distribution
#         action = tf.random.categorical(action_logits_t, 1)[0, 0]
#         action_probs_t = tf.nn.softmax(action_logits_t)

#         # Store critic values
#         values = values.write(t, tf.squeeze(value))

#         # Store log probability of the action chosen
#         action_probs = action_probs.write(t, action_probs_t[0, action]) # type: ignore

#         # Apply action to the environment to get next state and reward
#         state, reward, done = tf_env_step(action)
#         state.set_shape(initial_state_shape)

#         # Store reward
#         rewards = rewards.write(t, reward)

#         if tf.cast(done, tf.bool):
#             break
        

#     states = states.stack().mark_used() # type: ignore
#     action_probs = action_probs.stack().mark_used() # type: ignore
#     values = values.stack().mark_used() # type: ignore
#     rewards = rewards.stack().mark_used() # type: ignore

#     returns = get_expected_return(rewards, gamma=0.99)

#     # write to history
#     for i in range(num_episodes):
#         history[0].write(history_index, states)
#         history[1].write(history_index, action_probs[i])
#         history[2].write(history_index, values[i])
#         history[3].write(history_index, returns[i])
#         history_index = (history_index + 1) % history_size

#     return history, history_index


# def get_expected_return(
#     rewards: tf.Tensor, 
#     gamma: float, 
#     standardize: bool = True) -> tf.Tensor:
#     """Compute expected returns per timestep."""

#     n = tf.shape(rewards)[0]
#     returns = tf.TensorArray(dtype=tf.float32, size=n)

#     # Start from the end of `rewards` and accumulate reward sums
#     # into the `returns` array
#     rewards = tf.cast(rewards[::-1], dtype=tf.float32)
#     discounted_sum = tf.constant(0.0)
#     discounted_sum_shape = discounted_sum.shape
#     for i in tf.range(n):
#         reward = rewards[i]
#         discounted_sum = reward + gamma * discounted_sum
#         discounted_sum.set_shape(discounted_sum_shape)
#         returns = returns.write(i, discounted_sum)
#     returns = returns.stack()[::-1]

#     return returns

# huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

# def compute_loss(
#     action_probs: tf.Tensor,  
#     values: tf.Tensor,  
#     returns: tf.Tensor) -> tf.Tensor:
#     """Computes the combined Actor-Critic loss."""

#     advantage = returns - values

#     action_log_probs = tf.math.log(action_probs)
#     actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

#     critic_loss = huber_loss(values, returns)

#     return actor_loss + critic_loss


# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


# @tf.function
# def train_step(
#     initial_state: tf.Tensor, 
#     model: tf.keras.Model, 
#     optimizer: tf.keras.optimizers.Optimizer, 
#     gamma: float, 
#     max_steps_per_episode: int) -> tf.Tensor:
#     """Runs a model training step."""

#     with tf.GradientTape() as tape:
#         # Run the model for one episode to collect training data
#         action_probs, values, rewards = run_episode(initial_state, model, max_steps_per_episode) 

#         # Calculate the expected returns
#         returns = get_expected_return(rewards, gamma)

#         # Convert training data to appropriate TF tensor shapes
#         action_probs, values, returns = [
#             tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

#     # Calculate the loss values to update our network
#     loss = compute_loss(action_probs, values, returns)

#     # Compute the gradients from the loss
#     grads = tape.gradient(loss, model.trainable_variables)

#     # Apply the gradients to the model's parameters
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))

#     episode_reward = tf.math.reduce_sum(rewards)

#     return episode_reward
# @tf.function
# def run():
#     states_history = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
#     action_prob_history = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
#     value_history = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
#     returns_history = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

#     history_index = 0
#     history_size = 10000
#     max_steps_per_episode = 200
#     gamma = 0.95

#     for i in tf.range(1000):
#         current_state = tf_env_reset()

#         (states_history, action_prob_history, value_history, returns_history), history_index = \
#             run_episode(
#             current_state, 
#             model, 
#             max_steps_per_episode, 
#             (states_history, action_prob_history, value_history, returns_history), 
#             history_index, 
#             history_size)

#         # gather batch
#         batch_size = 100
#         idxs = tf.random.uniform(shape=(batch_size,), minval=0, maxval=min(states_history.size(), history_size), dtype=tf.int32)
#         states_batch = states_history.gather(idxs)
#         action_prob_batch = action_prob_history.gather(idxs)
#         value_batch = value_history.gather(idxs)
#         returns_batch = returns_history.gather(idxs)

#         tf.print(states_batch.shape, action_prob_batch.shape, value_batch.shape, returns_batch.shape)


#         tf.print(i, history_index)
        


# run()

