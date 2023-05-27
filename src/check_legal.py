import chess_engine
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

env = chess_engine.DiagonalChess()
env.reset()

np.set_printoptions(threshold=4096)

# mask = chess_engine.internal.get_legal_moves_mask(env.board, env.isBlack)
# print(mask)
# # generate random floats of size 4096
# action = np.random.rand(4096)
# print(action)
# # apply mask
# action = action * mask
# print(action)
# # get index of max value
# action = int(np.argmax(action))
# print(action)
# # check if action is legal

# obs, reward, done = env.step(action)

# print(reward)

# iterate 32 times and check if mask is correct
for i in range(32):
    mask = chess_engine.internal.get_legal_moves_mask(env.board, env.isBlack)
    action = np.random.rand(4096)
    action = action * mask
    action = int(np.argmax(action))
    obs, reward, done = env.step(action)
    print(reward)
    print(action)