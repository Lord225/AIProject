from typing import List
import gym
import numpy as np
import tensorflow as tf
import tqdm
import config_file
import tensorboard
from reinforce.data_collector import run_episode_and_get_history
from reinforce.replay_memory import ReplayMemory
from reinforce.train import training_step_dqnet_target_critic
import chess_engine

# This implements reinforcement learning for diagonal chess
# where other players is randomly choosing moves
# 
# it uses [6, 8, 8] input shape and 4096 actions
# 
# alternativly it can use [7, 8, 8] (with avalible moves) and 2x8x8 actions (move from, move to)

# env
env = chess_engine.DiagonalChess()
# env specs
input_shape = [8, 8, 6]
n_outputs = 4096

def env_step(action):
    _, reward1, done1 = env.step(action)
    action = np.random.randint(0, 4096)
    state2, reward2, done2 = env.step(action)

    reward = reward1
    done = done1 or done2

    return (state2.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action], [tf.float32, tf.float32, tf.int32]) #type: ignore


# model
def get_model():
    inputs = tf.keras.layers.Input(shape=input_shape)
    common = tf.keras.layers.Conv2D(32, 3, activation='elu', padding='same')(inputs) # 8 8
    common = tf.keras.layers.MaxPool2D()(common) # 2 2
    common = tf.keras.layers.Conv2D(64, 2, activation='elu', padding='same')(common) # 4 4 
    common = tf.keras.layers.MaxPool2D()(common) # 2 2
    common = tf.keras.layers.Conv2D(128, 2, activation='elu', padding='same')(common) # 4 4 
    common = tf.keras.layers.Flatten()(common)
    common = tf.keras.layers.Dense(1024, activation='elu')(common)
    common = tf.keras.layers.Dense(512, activation='elu')(common)

    Q_values = tf.keras.layers.Dense(n_outputs, activation='softmax')(common)

    critic = tf.keras.layers.Dense(256, activation='elu')(common)
    critic = tf.keras.layers.Dense(1)(common)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[Q_values, critic])

    model.summary()
    
    return model

actor_model = get_model()
target_model = get_model()
target_model.set_weights(actor_model.get_weights())

RUN_VERSION = "chess-v2.0"


train_summary_writer = tf.summary.create_file_writer(config_file.LOG_DIR+RUN_VERSION) #type: ignore

batch_size = 32
discount_rate = 0.99 # maybe 0.8 or even 0.5
episodes = 10000
train_iters_per_episode = 50
max_steps_per_episode = 25
target_update_freq = 10
minibatch_size = 32
replay_memory_size = 10000
save_freq = 250
lr = 3e-2

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

def run():
    replay_memory = ReplayMemory(replay_memory_size)

    t = tqdm.tqdm(range(episodes))
    for episode in t:
        # run episode
        state = env.reset()
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
    
            training_step_dqnet_target_critic(batch, minibatch_size, train_iters_per_episode, discount_rate, target_model, actor_model, optimizer, n_outputs)
        # update target network
        if episode % target_update_freq == 0:
            target_model.set_weights(actor_model.get_weights())

        # save model
        if episode % save_freq == 0:
            actor_model.save(f"{config_file.MODELS_DIR}actor_model_{RUN_VERSION}_{config_file.RUN_NAME}_{episode}.h5")


run()