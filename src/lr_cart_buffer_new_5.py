from typing import List
import gym
import numpy as np
import tensorflow as tf
import tqdm
import config_file
import tensorboard
from reinforce.data_collector import run_episode_and_get_history
from reinforce.replay_memory import ReplayMemory
from reinforce.train import training_step

# RL implementation 5
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
def env_step(action):
    state, reward, done, _, _ = env.step(action)
    return (state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action], [tf.float32, tf.float32, tf.int32]) #type: ignore


# networks
def get_network():
    inputs = tf.keras.layers.Input(shape=(4,))
    common = tf.keras.layers.Dense(32, 'elu')(inputs)
    common = tf.keras.layers.Dense(32, 'elu')(common)
    common = tf.keras.layers.Dense(32, 'elu')(common)
    common = tf.keras.layers.Dense(32, 'elu')(common)
    common = tf.keras.layers.Dense(32, 'elu')(common)
    common = tf.keras.layers.Dense(32, 'elu')(common)
    Q_values = tf.keras.layers.Dense(n_outputs)(common)
    critic = tf.keras.layers.Dense(1)(common)
    model = tf.keras.models.Model(inputs=[inputs], outputs=[Q_values, critic])

    return model

actor_model = get_network()
target_model = get_network()
target_model.set_weights(actor_model.get_weights())

RUN_VERSION = "v5.2"

train_summary_writer = tf.summary.create_file_writer(config_file.LOG_DIR+RUN_VERSION) #type: ignore

batch_size = 1024
discount_rate = 0.99
episodes = 1000
train_iters_per_episode = 20
max_steps_per_episode = 200
target_update_freq = 50
minibatch_size = 128
replay_memory_size = 50000
save_freq = 250
lr = 1e-3

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

def run():
    replay_memory = ReplayMemory(replay_memory_size)

    t = tqdm.tqdm(range(episodes))
    for episode in t:
        # run episode
        state, _ = env.reset()
        state = tf.constant(state, dtype=tf.float32)
        (states, action_probs, returns, next_states, dones), total_rewards = run_episode_and_get_history(state, actor_model, max_steps_per_episode, discount_rate, tf_env_step)  # type: ignore

        # add to replay memory
        replay_memory.add(states, action_probs, returns, next_states, dones)

        # log
        with train_summary_writer.as_default():
            tf.summary.scalar('reward', total_rewards, step=episode)

        t.set_description(f"Episode {episode} - Reward: {total_rewards:.2f}")
        
        # train
        if len(replay_memory) > batch_size+1:
            batch = replay_memory.sample(batch_size)
    
            training_step(batch, minibatch_size, train_iters_per_episode, discount_rate, target_model, actor_model, optimizer, n_outputs)
        # update target network
        if episode % target_update_freq == 0:
            target_model.set_weights(actor_model.get_weights())

        # save model
        if episode % save_freq == 0:
            actor_model.save(f"{config_file.MODELS_DIR}actor_model_{RUN_VERSION}_{config_file.RUN_NAME}_{episode}.h5")


run()