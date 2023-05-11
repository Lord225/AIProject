import chess_engine
import collections
import gym
import numpy as np
import tensorflow as tf
import tqdm
from typing import Any, List, Sequence, Tuple
import config_file

# Create the environment
env = chess_engine.DiagonalChess()


def get_network():
    inputs = tf.keras.Input(shape=(8, 8, 8))
    x = tf.keras.layers.Conv2D(32,  3, activation="relu", padding='same')(inputs)
    x = tf.keras.layers.Conv2D(64,  3, padding='same')(x)
    x = tf.keras.layers.Conv2D(128,  2, padding='same')(x)
    x = tf.keras.layers.Conv2D(128,  2, padding='same')(x)
    x = tf.keras.layers.Conv2D(64,  2, padding='same')(x)
    actor = tf.keras.layers.Flatten()(x)
    critic = tf.keras.layers.Flatten()(x)
    critic = tf.keras.layers.Dense(2)(critic)
    critic = tf.keras.layers.Dense(1)(critic)

    model = tf.keras.Model(inputs=inputs, outputs=[actor, critic])

    print(model.summary())

    return model

model = get_network()

def env_step(action):
    state1, reward1, done1 = env.step(int(action))

    random_action = np.random.randint(0, 4096)
    state2, reward2, done2 = env.step(int(random_action))

    state = state2
    reward = reward1 #- max(reward2, 0)
    done = done1 or done2

    return (state1.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action], [tf.float32, tf.float32, tf.int32]) # type: ignore


def run_episode(
    initial_state: tf.Tensor,  
    model: tf.keras.Model, 
    max_steps: int
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Runs a single episode to collect training data."""

  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state

  for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    state = tf.expand_dims(state, 0)

    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state) # type: ignore
    action_probs_t = tf.nn.softmax(action_logits_t)

    # Sample next action from the action probability distribution
    action = tf.random.categorical(action_probs_t, 1)[0, 0]

    # Store critic values
    values = values.write(t, tf.squeeze(value))

    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0, action]) # type: ignore

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

  print('a', action_probs.numpy())
  print('b', values.numpy())
  print('c', rewards.numpy())

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

  return returns


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
loss_metric = tf.keras.metrics.Mean(name='loss')

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


#@tf.function
def train_step(
    initial_state: tf.Tensor, 
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer, 
    gamma: float,
    max_steps_per_episode: int) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:

    # Run the model for one episode to collect training data
    action_probs, values, rewards = run_episode(initial_state, model, max_steps_per_episode) 

    # Calculate the expected returns
    returns = get_expected_return(rewards, gamma)

    # Convert training data to appropriate TF tensor shapes
    action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

    # Calculate the loss values to update our network
    loss = compute_loss(action_probs, values, returns)

    loss_metric(loss)

  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)

  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward

discount_rate = 0.95
episodes = 200000
save_freq = 250
lr = 1e-3
max_steps_per_episode = 20

RUN_VERSION = "v5.0"

train_summary_writer = tf.summary.create_file_writer(config_file.LOG_DIR+RUN_VERSION) #type: ignore

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

running_avg = collections.deque(maxlen=1000)

t = tqdm.trange(episodes)
for i in t:
    epsilon = max(0.01, 1 - i/1000)
    initial_state = env.reset()
    initial_state = tf.constant(initial_state, dtype=tf.float32)
    episode_reward = float(train_step(initial_state, model, optimizer, discount_rate, max_steps_per_episode))


    running_avg.append(episode_reward)
    avg = sum(running_avg)/len(running_avg)
    with train_summary_writer.as_default():
        tf.summary.scalar('reward', episode_reward, step=i)
        tf.summary.scalar('reward_avg', avg, step=i)
        tf.summary.scalar('loss', loss_metric.result(), step=i)
        loss_metric.reset_states()
    t.set_postfix(episode_reward=episode_reward, avg=avg)
  


    if i % save_freq == 0:
        model.save(f"{config_file.MODELS_DIR}chess_{RUN_VERSION}_{config_file.RUN_NAME}_{i}.h5")
