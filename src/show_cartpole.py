import gym
import numpy as np
import tensorflow as tf
import config_file
import argparse

# get model name from command line
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model name")
parser.add_argument("--save", help="save as gif", action="store_false")
args = parser.parse_args()

# get model name
model_name = args.model

# get save gif
save_gif = args.save
render_for_human = not save_gif

# load model
model = tf.keras.models.load_model(f"{config_file.MODELS_DIR}/{model_name}")

# load environment, human
env = gym.make("CartPole-v1", render_mode="rgb_array_list" if save_gif else "human")


# play forever
while True:
    # reset environment
    state, info = env.reset()
    # play until done
    for _ in range(200):
        # render environment
        if render_for_human:
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

    if save_gif:
        print("Saving gif...")
        imgs = env.render()
        # save gif
        import imageio
        imageio.mimsave(f"{config_file.RETAIN_DIR}/{model_name}.gif", imgs) # type: ignore

        exit()
    
    # close environment
    env.close()
