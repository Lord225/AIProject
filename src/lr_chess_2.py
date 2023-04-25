import chess_engine


env = chess_engine.DiagonalChess()

env.reset()
for i in range(1000):
    env.