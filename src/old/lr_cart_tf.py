from typing import List, Tuple
import gym
import numpy as np
import tensorflow as tf
import tqdm
import config_file

# works fine but do not have buffer

# nie da się dodać bufora praktycznie

env = gym.make("CartPole-v1")


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(32, input_shape=(4,), activation="elu"))
model.add(tf.keras.layers.Dense(32, activation="elu"))
model.add(tf.keras.layers.Dense(2))

def env_step(action):
    state, reward, done, _, _ = env.step(action)
    return (state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action], [tf.float32, tf.float32, tf.int32]) # type: ignore


def run_episode(
        initial_state: tf.Tensor,
        model: tf.keras.Model, 
        max_steps: int) -> Tuple[tf.Tensor, tf.Tensor]:
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        action_logits_t = model(state)

        # Sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

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
    rewards = rewards.stack()

    return action_probs, rewards

def get_expected_return(
    rewards: List, 
    gamma: float):
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
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
        action_probs, rewards = run_episode(
            initial_state, model, max_steps_per_episode) 

        # Calculate the expected returns
        returns = get_expected_return(rewards, gamma) # type: ignore

        # Convert training data to appropriate TF tensor shapes
        action_probs, returns = [
            tf.expand_dims(x, 1) for x in [action_probs, returns]] 

        # Calculate the loss values to update our network
        loss = -tf.math.reduce_sum(tf.math.log(action_probs) * returns)

        train_loss(loss)

    # Compute the gradients from the loss
    grads = tape.gradient(loss, model.trainable_variables)

    # Apply the gradients to the model's parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward



train_loss = tf.keras.metrics.Mean(name='train_loss')
train_summary_writer = tf.summary.create_file_writer(config_file.LOG_DIR+"tfv1") # type: ignore

for i in tqdm.tqdm(range(1000)):
    initial_state, info = env.reset()
    initial_state = tf.constant(initial_state, dtype=tf.float32)
    rewards = train_step(initial_state, model, optimizer, 0.99, 200)

    with train_summary_writer.as_default():
        tf.summary.scalar('reward', rewards, step=i)
        tf.summary.scalar('loss', train_loss.result(), step=i)
        train_loss.reset_states()


