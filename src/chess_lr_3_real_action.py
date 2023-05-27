import chess_engine
from typing import List
import gym
import numpy as np
import tensorflow as tf
import tqdm
import config_file
import tensorboard
from collections import deque
from reinforce.data_collector import run_episode_and_get_history_4
from reinforce.replay_memory import ReplayMemory, ReplayMemory2
from reinforce.train import training_step_dqnet_target_critic, training_step_no_critic_no_target
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--resume", type=str, default=None, help="resume from a model") 

args = parser.parse_args()

env = chess_engine.DiagonalChess()
n_outputs = 4096

def env_step(action):
    state1, reward1, done1 = env.step(int(action))

    random_action = np.random.randint(0, 4096)
    state2, reward2, done2 = env.step(int(random_action))

    state = state2
    reward = reward1 - (1 if reward2 > 2 else 0)
    done = done1 or done2

    return (state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action], [tf.float32, tf.float32, tf.int32]) #type: ignore

def random_legal_action():
    move = chess_engine.internal.random_legal_move(env.board, False)
    if move is None:
        return 0
    else:
        return chess_engine.internal.move_to_int(*move)


def tf_random_legal_action():
    return tf.numpy_function(random_legal_action, [], tf.int32) #type: ignore


def get_network() -> tf.keras.Model:
    if args.resume is not None:
        model = tf.keras.models.load_model(args.resume)
        print("loaded model from", args.resume)
        return model # type: ignore
    
    inputs = tf.keras.Input(shape=(8, 8, 8))
    x = tf.keras.layers.Conv2D(32,  3, activation="elu", padding='same')(inputs)
    x = tf.keras.layers.Conv2D(64,  3, padding='same', activation="elu")(x)
    x = tf.keras.layers.Conv2D(128,  2, padding='same', activation="elu")(x)
    x = tf.keras.layers.Conv2D(256,  2, padding='same', activation="elu")(x)
    x = tf.keras.layers.Conv2D(256,  2, padding='same', activation="elu")(x)
    x = tf.keras.layers.Conv2D(256,  2, padding='same', activation="elu")(x)
    x = tf.keras.layers.Conv2D(128,  2, padding='same', activation="elu")(x)
    x = tf.keras.layers.Conv2D(64,  4, padding='same', activation="elu")(x)
    x = tf.keras.layers.Conv2D(64,  4, padding='same', activation="elu")(x)
    actor = tf.keras.layers.Flatten()(x)
    critic = tf.keras.layers.Flatten()(x)
    critic = tf.keras.layers.Dense(2, activation="relu")(critic)
    critic = tf.keras.layers.Dense(1)(critic)

    model = tf.keras.Model(inputs=inputs, outputs=[actor, critic])

    model.compile()

    print(model.summary())

    return model

actor_model = get_network()
target_model = get_network()
target_model.set_weights(actor_model.get_weights())

RUN_VERSION = "v4.0"

print("run:", config_file.LOG_DIR+RUN_VERSION)
train_summary_writer = tf.summary.create_file_writer(config_file.LOG_DIR+RUN_VERSION) #type: ignore

batch_size = 1024
discount_rate = 0.1
episodes = 200000
minibatch_size = 128
train_iters_per_episode = 10
max_steps_per_episode = 15
target_update_freq = 500
replay_memory_size = 50000
save_freq = 250

eps_decay_len = 100
eps_min = 0.1


lr = 2.5e-4

optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=lr,
                rho=0.95,
                momentum=0.0,
                epsilon=1e-07,
                centered=True,
            )

# optimizer = tf.keras.optimizers.Adam(
#     learning_rate=lr,
#     beta_1=0.85,
#     beta_2=0.95,    
# )

running_avg = deque(maxlen=500)

def run():
    replay_memory = ReplayMemory2(replay_memory_size, (replay_memory_size, 8, 8, 8))

    t = tqdm.tqdm(range(episodes))
    for episode in t:
        epsilon = max(1 - episode / eps_decay_len, eps_min)

        # run episode
        state = env.reset()
        state = tf.constant(state)
        (states, action_probs, returns, next_states, dones), total_rewards = run_episode_and_get_history_4(state, 
                                                                                                           actor_model, 
                                                                                                           max_steps_per_episode, 
                                                                                                           discount_rate,
                                                                                                           epsilon,
                                                                                                           tf_env_step
                                                                                                           )  # type: ignore

        # add to replay memory
        replay_memory.add(states, action_probs, returns, next_states, dones)
        running_avg.append(total_rewards)
        avg = sum(running_avg)/len(running_avg)



        # log
        with train_summary_writer.as_default():
            tf.summary.scalar('reward', total_rewards, step=episode)
            tf.summary.scalar('reward_avg', avg, step=episode)
            tf.summary.scalar('lenght', states.shape[0], step=episode)

        t.set_description(f"Episode {episode} - Reward: {total_rewards:.2f} - Avg: {avg:.2f}")
        
        #train
        if len(replay_memory) > batch_size+1:
            batch = replay_memory.sample(batch_size)
    
            training_step_dqnet_target_critic(batch, 
                                              minibatch_size, 
                                              train_iters_per_episode, 
                                              discount_rate, 
                                              target_model, 
                                              actor_model, 
                                              optimizer,
                                              n_outputs,
                                              )
            
        # update target network
        if episode % target_update_freq == 0:
            target_model.set_weights(actor_model.get_weights())

        # save model
        if episode % save_freq == 0:
            actor_model.save(f"{config_file.MODELS_DIR}chess_{RUN_VERSION}_{config_file.RUN_NAME}_{episode}.h5")


run()