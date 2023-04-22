from typing import Tuple
import unittest
import numpy as np

from chess_engine.diagchess import fen_to_svg, generate_start_board, piece_to_fen, to_fen


class DiagonalChess:
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=np.int8)
        self.reset()


    def reset(self):
        """
        resets the board to the starting position
        """
        
        self.board = generate_start_board()
    

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, int, bool]:
        """
        ## returns
        - observation: np.ndarray
        - reward: int
        - done: bool


        ### Observations
        state is represent by an 8x8x8 array
        Plane 0 represents pawns
        Plane 1 represents rooks
        Plane 2 represents knights
        Plane 3 represents bishops
        Plane 4 represents queens
        Plane 5 represents kings

        #### Example
        ```py
        board.layer_board[0,::-1,:].astype(int)
        array([[ 0,  0,  0,  0,  0,  0,  0,  0],
               [-1, -1, -1, -1, -1, -1, -1, -1],
               [ 0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0],
               [ 1,  1,  1,  1,  1,  1,  1,  1],
               [ 0,  0,  0,  0,  0,  0,  0,  0]])
        ```

        ### Actions
        Its action space consist of 64x64=4096 actions:
        There are 8x8 = 64 piece from where a piece can be picked up
        And another 64 pieces from where a piece can be dropped.

        [Policy on deciding how to treat illegal moves](https://ai.stackexchange.com/questions/7979/why-does-the-policy-network-in-alphazero-work)

        Simplest way is to implement an method that returns a array of legal moves for each piece.

        Ideas to check:
        * add a small penalty to the reward for illegal moves
        * add small reward for legal moves
        * use 4 numbers to indicate the move (from, to, promote to, promote from) instead of 8x8x8x8 array of probabilities
        * mask probabilities of illegal moves to 0 and use max to select the move

        """

        raise NotImplementedError

    def render(self):
        """
        Should render the board using the python-chess library
        """

        return fen_to_svg(to_fen(self.board))
    
    def __str__(self):
        output = ''
        for row in self.board:
            output += f"{' '.join([piece_to_fen(piece) for piece in row])} \n"
        return output
    
    def __repr__(self):
        return to_fen(self.board)




