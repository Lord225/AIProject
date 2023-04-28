import chess_engine
from typing import List
import gym
import numpy as np
import tensorflow as tf
import tqdm
import config_file
import tensorboard

from reinforce.data_collector import run_episode_and_get_history_3
from reinforce.replay_memory import ReplayMemory
from reinforce.train import training_step_dqnet_target_critic_state_transform
env = chess_engine.DiagonalChess()
n_outputs = 4096

def transform_state(board_state):
    return chess_engine.internal.board_to_observation(board_state)

def tf_transform_state(board_state):
    return tf.numpy_function(transform_state, [board_state], [tf.float32])

def transform_state_batch(board_state):
    board_state = board_state.astype(np.int8)
    return chess_engine.internal.board_to_observation_batch(board_state)

def tf_transform_state_batch(board_state):
    return tf.numpy_function(transform_state_batch, [board_state], [tf.float32])

def env_step(action):
    state1, reward1, done1 = env.step_board_obs(int(action))


    random_action = np.random.randint(0, 4096)
    state2, reward2, done2 = env.step_board_obs(int(random_action))


    state = state2
    reward = reward1 #- reward2
    done = done1 or done2


    return (state.astype(np.int8), np.array(reward, np.float32), np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action], [tf.int8, tf.float32, tf.int32]) #type: ignore



def get_network():
    inputs = tf.keras.Input(shape=(8, 8, 8))
    x = tf.keras.layers.Conv2D(32, 3, activation="elu", padding='same')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation="elu", padding='same')(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="elu", padding='same')(x)
    x = tf.keras.layers.Conv2D(64, 2, activation="elu", padding='same')(x)
    x = tf.keras.layers.Conv2D(64, 2, activation="elu", padding='same')(x)
    actor = tf.keras.layers.Flatten()(x)
    critic = tf.keras.layers.Flatten()(x)
    critic = tf.keras.layers.Dense(2, activation="elu")(critic)
    critic = tf.keras.layers.Dense(1)(critic)

    model = tf.keras.Model(inputs=inputs, outputs=[actor, critic])

    print(model.summary())

    return model

actor_model = get_network()
target_model = get_network()
target_model.set_weights(actor_model.get_weights())

RUN_VERSION = "v3.0"

train_summary_writer = tf.summary.create_file_writer(config_file.LOG_DIR+RUN_VERSION) #type: ignore

batch_size = 1024
discount_rate = 0.99
episodes = 20000
train_iters_per_episode = 10
max_steps_per_episode = 10
target_update_freq = 100
minibatch_size = 128
replay_memory_size = 4000
save_freq = 250
lr = 3e-2

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

def run():
    replay_memory = ReplayMemory(replay_memory_size)

    t = tqdm.tqdm(range(episodes))
    for episode in t:
        epsilon = max(1 - episode / 1000, 0.01)

        # run episode
        state = env.reset_board()
        state = tf.constant(state)
        (states, action_probs, returns, next_states, dones), total_rewards = run_episode_and_get_history_3(state, 
                                                                                                           actor_model, 
                                                                                                           max_steps_per_episode, 
                                                                                                           discount_rate,
                                                                                                           epsilon,
                                                                                                           tf_env_step, 
                                                                                                           tf_transform_state)  # type: ignore

        # add to replay memory
        replay_memory.add(states, action_probs, returns, next_states, dones)

        # log
        with train_summary_writer.as_default():
            tf.summary.scalar('reward', total_rewards, step=episode)
            tf.summary.scalar('lenght', states.shape[0], step=episode)

        t.set_description(f"Episode {episode} - Reward: {total_rewards:.2f}")
        
        # train
        if len(replay_memory) > batch_size+1:
            batch = replay_memory.sample(batch_size)
    
            training_step_dqnet_target_critic_state_transform(batch, 
                                              minibatch_size, 
                                              train_iters_per_episode, 
                                              discount_rate, 
                                              target_model, 
                                              actor_model, 
                                              optimizer,
                                              n_outputs,
                                              tf_transform_state_batch,
                                              )
        # update target network
        if episode % target_update_freq == 0:
            target_model.set_weights(actor_model.get_weights())

        # save model
        if episode % save_freq == 0:
            actor_model.save(f"{config_file.MODELS_DIR}chess_{RUN_VERSION}_{config_file.RUN_NAME}_{episode}.h5")


run()