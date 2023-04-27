from typing import List
import gym
import numpy as np
import tensorflow as tf
import tqdm
import config_file
import tensorboard
# argpars
import argparse

# get model name from command line
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model name")
args = parser.parse_args()

# get model name
model_name = args.model

# load model
model = tf.keras.models.load_model(f"{config_file.MODELS_DIR}/{model_name}")

# load environment, human
env = gym.make("CartPole-v1", render_mode="human")


# play forever
while True:
    # reset environment
    state, info = env.reset()
    # play until done
    while True:
        # render environment
        env.render()
        # predict action
        Q_values = model.predict(state[np.newaxis], verbose=0) #type: ignore
        action = np.argmax(Q_values[0])
        # step environment
        next_state, reward, done, *info = env.step(action)
        # update state
        state = next_state
        # check if done
        if done:
            break
    # close environment
    env.close()





