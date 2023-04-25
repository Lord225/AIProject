import chess_engine
from typing import List
import gym
import numpy as np
import tensorflow as tf
import tqdm
import config_file
import tensorboard

from reinforce.data_collector import run_episode_custom_action

env = chess_engine.DiagonalChess()

def action_transform(action_logits):
    return chess_engine.internal.array_action_to_move_vectorized(env.board, action_logits, env.isBlack) 

def tf_action_transform(action_logits):
    return tf.numpy_function(action_transform, [action_logits], [tf.int32])

def transform_state(board_state):
    return chess_engine.internal.board_to_observation(board_state)

def tf_transform_state(board_state):
    return tf.numpy_function(transform_state, [board_state], [tf.float32])


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
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding='same')(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding='same')(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding='same')(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding='same')(x)
    actor = tf.keras.layers.Conv2D(2, 2, activation="relu", padding='same')(x)
    critic = tf.keras.layers.Flatten()(x)
    critic = tf.keras.layers.Dense(1)(critic)

    model = tf.keras.Model(inputs=inputs, outputs=[actor, critic])

    print(model.summary())

    return model

actor_model = get_network()
target_model = get_network()

t = tqdm.tqdm(range(1000))
for i in t:
    initial_state = env.reset()
    initial_state = tf.constant(initial_state)
    outputs = run_episode_custom_action(
        initial_state,
        actor_model,
        100,
        tf_env_step,
        tf_action_transform,
        tf_transform_state,
    ) # type: ignore

    t.set_description(f"Episode {i}")

