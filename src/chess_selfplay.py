import chess_engine
from typing import List
import gym
import numpy as np
import tensorflow as tf
import tqdm
import config_file
import tensorboard
from collections import deque
from reinforce.data_collector import run_episode_and_get_history_selfplay
from reinforce.replay_memory import ReplayMemory2
from reinforce.train import training_step_dqnet_target_critic, training_step_dqnet_target_critic_2
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--white", type=str, default=None, help="resume white") 
parser.add_argument("--black", type=str, default=None, help="resume black")

args = parser.parse_args()

env = chess_engine.DiagonalChess()
n_outputs = 4096

def env_step(action):
    state, reward, done = env.step(int(action))
 
    return (state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action], [tf.float32, tf.float32, tf.int32]) #type: ignore

def moves_mask():
    return chess_engine.internal.get_legal_moves_mask(env.board, env.isBlack).astype(np.float32)

def tf_moves_mask():
    return tf.numpy_function(moves_mask, [], tf.float32) #type: ignore



def get_network(isBlack: bool) -> tf.keras.Model:
    if isBlack and args.black is not None:
        model = tf.keras.models.load_model(args.black)
        print("loaded black model from", args.black)
        return model # type: ignore
    elif (not isBlack) and args.white is not None:
        model = tf.keras.models.load_model(args.white)
        print("loaded white model from", args.white)
        return model # type: ignore
    
    inputs = tf.keras.Input(shape=(8, 8, 8))
    x = tf.keras.layers.Conv2D(32,  3, activation="elu", padding='same')(inputs)
    x = tf.keras.layers.Conv2D(128,  3, padding='same', activation="elu")(x)
    x = tf.keras.layers.Conv2D(128,  2, padding='same', activation="elu")(x)
    x = tf.keras.layers.Conv2D(128,  2, padding='same', activation="elu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation="elu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    actor =  tf.keras.layers.Dense(4096)(x)
    critic = tf.keras.layers.Dense(2, activation="relu")(x)
    critic = tf.keras.layers.Dense(1)(critic)

    model = tf.keras.Model(inputs=inputs, outputs=[actor, critic])

    model.compile()

    print(model.summary())

    return model



actor_model_white = get_network(False)
target_model_white = get_network(False)
target_model_white.set_weights(actor_model_white.get_weights())

actor_model_black = get_network(True)
target_model_black = get_network(True)
target_model_black.set_weights(actor_model_black.get_weights())


RUN_VERSION = "_selfplay"

print("run:", config_file.LOG_DIR+RUN_VERSION)
train_summary_writer = tf.summary.create_file_writer(config_file.LOG_DIR+RUN_VERSION) #type: ignore

batch_size = 2048
discount_rate = 0.99
episodes = 1000000
minibatch_size = 256
train_iters_per_episode = 8
train_interval = 1
max_steps_per_episode = 20
target_update_freq = 300
replay_memory_size = 15_000
save_freq = 1000

eps_decay_len = 100
eps_min = 0.05

lr = 7e-6

optimizer_white = tf.keras.optimizers.Adam(
    learning_rate=lr,
    beta_1=0.80,
    beta_2=0.90,    
)

optimizer_black = tf.keras.optimizers.Adam(
    learning_rate=lr,
    beta_1=0.80,
    beta_2=0.90,    
)


running_avg_black = deque(maxlen=500)
running_avg_white = deque(maxlen=500)


def run():
    replay_memory_white = ReplayMemory2(replay_memory_size, (replay_memory_size, 8, 8, 8))
    replay_memory_black = ReplayMemory2(replay_memory_size, (replay_memory_size, 8, 8, 8))

    t = tqdm.tqdm(range(episodes))
    for episode in t:
        epsilon = max(1 - episode / eps_decay_len, eps_min)

        state = env.reset()
        state = tf.constant(state)

        history_white, history_black, rewards_white, rewards_black = run_episode_and_get_history_selfplay(
            state,
            actor_model_white,
            actor_model_black,
            max_steps_per_episode,
            epsilon,
            epsilon,
            tf_env_step,
            tf_moves_mask,
        )

        replay_memory_white.add(*history_white)
        replay_memory_black.add(*history_black)
        running_avg_white.append(rewards_white)
        running_avg_black.append(rewards_black)

        avg_w = sum(running_avg_white) / len(running_avg_white)
        avg_b = sum(running_avg_black) / len(running_avg_black)

        with train_summary_writer.as_default():
            tf.summary.scalar("reward_avg_white", avg_w, step=episode)
            tf.summary.scalar("reward_avg_black", avg_b, step=episode)
            tf.summary.scalar("reward_white", rewards_white, step=episode)
            tf.summary.scalar("reward_black", rewards_black, step=episode)

        t.set_description(f"white: {avg_w:.2f} black: {avg_b:.2f} epsilon: {epsilon:.2f} std {np.std(running_avg_white):.2f}") # type: ignore

        if len(replay_memory_white) > batch_size+1 and episode % train_interval == 0:
            # train white
            batch_white = replay_memory_white.sample(batch_size)

            training_step_dqnet_target_critic(
                batch_white, 
                minibatch_size, 
                train_iters_per_episode, 
                discount_rate, 
                target_model_white, 
                actor_model_white, 
                optimizer_white,
                n_outputs,
            )

            #train black
            batch_black = replay_memory_black.sample(batch_size)

            training_step_dqnet_target_critic_2(
                batch_black, 
                minibatch_size,
                train_iters_per_episode,
                discount_rate,
                target_model_black,
                actor_model_black,
                optimizer_white,
                n_outputs,
            )

        if episode % target_update_freq == 0:
            target_model_white.set_weights(actor_model_white.get_weights())
            target_model_black.set_weights(actor_model_black.get_weights())
        
        if episode % save_freq == 0:
            actor_model_white.save(f"{config_file.MODELS_DIR}chess_white_{RUN_VERSION}_{config_file.RUN_NAME}_{episode}.h5")
            actor_model_black.save(f"{config_file.MODELS_DIR}chess_black_{RUN_VERSION}_{config_file.RUN_NAME}_{episode}.h5")


run()