import chess_engine
from typing import List
import gym
import numpy as np
import tensorflow as tf
import tqdm
import config_file
import tensorboard

from reinforce.data_collector import run_episode_and_get_history_2, run_episode_custom_action
from reinforce.replay_memory import ReplayMemory
from reinforce.train import training_step_dqnet_target_critic_custom_action

env = chess_engine.DiagonalChess()

def action_transform(action_logits):
    return chess_engine.internal.array_action_to_move_vectorized_one_board(env.board, action_logits, env.isBlack) 

def tf_action_transform(action_logits):
    return tf.numpy_function(action_transform, [action_logits], [tf.int32])

def transform_state(board_state):
    return chess_engine.internal.board_to_observation(board_state)

def tf_transform_state(board_state):
    return tf.numpy_function(transform_state, [board_state], [tf.float32])

def transform_state_batch(board_state):
    board_state = board_state.astype(np.int8)
    return chess_engine.internal.board_to_observation_batch(board_state)

def tf_transform_state_batch(board_state):
    return tf.numpy_function(transform_state_batch, [board_state], [tf.float32])

def transform_action(state, action):
    state = state.astype(np.int8)
    return chess_engine.internal.array_action_to_move_vectorized(state, action, False) 

def tf_transform_action(action, state):
    return tf.numpy_function(transform_action, [action, state], [tf.int32])

def env_step(action):
    state1, reward1, done1 = env.step_board_obs(int(action[0]))


    random_action = np.random.randint(0, 4096)
    state2, reward2, done2 = env.step_board_obs(int(random_action))

    state = state2
    reward = reward1 #- reward2
    done = done1 or done2


    return (state.astype(np.int8), np.array(reward, np.float32), np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action], [tf.int8, tf.float32, tf.int32]) #type: ignore


def test_obs():
    # obs = env.reset()
    # print("obs", obs)

    # sample_obs = tf.constant(obs)
    # sample_obs = tf_transform_state(sample_obs)
    # print("sample_obs", sample_obs)
    # # test if action will be generated correctly, random action of shape 2,8,8
    # test_action = tf.random.uniform([8, 8, 2], minval=0, maxval=1, dtype=tf.float32)

    # print("test_action", test_action)

    # test_action = tf_action_transform(test_action)

    # print("test_action", test_action)

    # obs, rew, done = env_step(test_action)

    # print("obs", obs)
    pass

def get_network():
    inputs = tf.keras.Input(shape=(8, 8, 8))
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding='same')(inputs)
    actor = tf.keras.layers.Conv2D(2, 2, activation="relu", padding='same')(x)
    critic = tf.keras.layers.Flatten()(x)
    critic = tf.keras.layers.Dense(1)(critic)

    model = tf.keras.Model(inputs=inputs, outputs=[actor, critic])

    print(model.summary())

    return model

actor_model = get_network()
target_model = get_network()
target_model.set_weights(actor_model.get_weights())

RUN_VERSION = "v3.0"

train_summary_writer = tf.summary.create_file_writer(config_file.LOG_DIR+RUN_VERSION) #type: ignore

batch_size = 32
discount_rate = 0.99
episodes = 5000
train_iters_per_episode = 6
max_steps_per_episode = 10
target_update_freq = 50
minibatch_size = 32
replay_memory_size = 1000
save_freq = 250
lr = 7e-2

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

buffer = ReplayMemory(replay_memory_size)
t = tqdm.tqdm(range(episodes))
for episode in t:
    initial_state = env.reset()
    initial_state = tf.constant(initial_state)
    (states, action_probs, returns, next_states, dones), total_rewards = run_episode_and_get_history_2(
        initial_state,
        actor_model,
        max_steps_per_episode,
        discount_rate,
        tf_env_step,
        tf_action_transform,
        tf_transform_state,
    ) # type: ignore


    with train_summary_writer.as_default():
        tf.summary.scalar('reward', total_rewards, step=episode)
    
    buffer.add(states, action_probs, returns, next_states, dones)

    t.set_description(f"Episode {episode}, {total_rewards:0.2f} {len(buffer)}")

    if len(buffer) > batch_size+1:
        batch = buffer.sample(batch_size)

        training_step_dqnet_target_critic_custom_action(batch, 
                                                        minibatch_size, 
                                                        train_iters_per_episode, 
                                                        discount_rate, 
                                                        target_model, 
                                                        actor_model, 
                                                        optimizer,
                                                        tf_transform_state_batch,
                                                        tf_transform_action,
                                                        )
    
    # update target network
    if episode % target_update_freq == 0:
        target_model.set_weights(actor_model.get_weights())

    # save model
    if episode % save_freq == 0:
        actor_model.save(f"{config_file.MODELS_DIR}actor_model_{RUN_VERSION}_{config_file.RUN_NAME}_{episode}.h5")
