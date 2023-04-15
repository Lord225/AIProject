from typing import Tuple
import numpy as np
import numba as nb
import chess
import chess.svg

import unittest


@nb.njit('int8(types.unicode_type)',cache=True)
def pice(name: str) -> int:
    PICES = {
        'PAWN': 1,
        'pawn': -1,
        'ROOK': 2,
        'rook': -2,
        'KNIGHT': 3,
        'knight': -3,
        'BISHOP': 4,
        'bishop': -4,
        'QUEEN': 5,
        'queen': -5,
        'KING': 6,
        'king': -6,
    }

    return PICES[name]

@nb.njit(cache=True)
def pice_to_fen(piece: int) -> str:
    pice_map = {
        1: 'p',
        -1: 'P',
        2: 'r',
        -2: 'R',
        3: 'n',
        -3: 'N',
        4: 'b',
        -4: 'B',
        5: 'q',
        -5: 'Q',
        6: 'k',
        -6: 'K',
        0: ' ',
    }

    return pice_map[piece]

@nb.njit('types.unicode_type(int8[:,:])', cache=True)
def to_fen(board: np.ndarray) -> str:
    """
    converts the board to a fen string
    """
    
    fen = ''
    for row in board:
        empty = 0
        for piece in row:
            if piece == 0:
                empty += 1
            else:
                if empty > 0:
                    fen += str(empty)
                    empty = 0
                fen += pice_to_fen(piece)
        if empty > 0:
            fen += str(empty)
        fen += '/'
    fen = fen[:-1]
    return fen

@nb.njit('int8[:,:]()',cache=True)
def generate_start_board() -> np.ndarray:
    board = np.zeros((8, 8), dtype=np.int8)

    board[0, :] = np.array([0             , 0             , 0             , pice('PAWN'), pice('ROOK'), pice('BISHOP'), pice('KNIGHT'), pice('KING')  ], dtype=np.int8)
    board[1, :] = np.array([0             , 0             , 0             , 0           , pice('PAWN'), pice('PAWN')  , pice('QUEEN') , pice('KNIGHT')], dtype=np.int8)
    board[2, :] = np.array([0             , 0             , 0             , 0           , 0           , pice('PAWN')  , pice('PAWN')  , pice('BISHOP')], dtype=np.int8)
    board[3, :] = np.array([pice('pawn')  , 0             , 0             , 0           , 0           , 0             , pice('PAWN')  , pice('ROOK')  ], dtype=np.int8)
    board[4, :] = np.array([pice('rook')  , pice('pawn')  , 0             , 0           , 0           , 0             , 0             , pice('PAWN')  ], dtype=np.int8)
    board[5, :] = np.array([pice('bishop'), pice('pawn')  , pice('pawn')  , 0           , 0           , 0             , 0             , 0             ], dtype=np.int8)
    board[6, :] = np.array([pice('knight'), pice('queen') , pice('pawn')  , pice('pawn'), 0           , 0             , 0             , 0             ], dtype=np.int8)
    board[7, :] = np.array([pice('king')  , pice('knight'), pice('bishop'), pice('rook'), pice('pawn'), 0             , 0             , 0             ], dtype=np.int8)
    return board


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
        Plane 6 represents 1/fullmove number (needed for markov property)
        Plane 7 represents can-claim-draw

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

        return chess.svg.board(chess.Board(to_fen(self.board)), size=500)
    
    def __str__(self):
        output = ''
        for row in self.board:
            output += f"{' '.join([pice_to_fen(pice) for pice in row])} \n"
        return output
    
    def __repr__(self):
        return to_fen(self.board)
        

class TestDiagonalChess(unittest.TestCase):
    def test_pice_to_unit(self):
        self.assertEqual(pice_to_fen(1), 'p')
        self.assertEqual(pice_to_fen(-1), 'P')
        self.assertEqual(pice_to_fen(2), 'r')
        self.assertEqual(pice_to_fen(-2), 'R')
        self.assertEqual(pice_to_fen(3), 'n')
        self.assertEqual(pice_to_fen(-3), 'N')
        self.assertEqual(pice_to_fen(4), 'b')
        self.assertEqual(pice_to_fen(-4), 'B')
        self.assertEqual(pice_to_fen(5), 'q')
        self.assertEqual(pice_to_fen(-5), 'Q')
        self.assertEqual(pice_to_fen(6), 'k')
        self.assertEqual(pice_to_fen(-6), 'K')
    
    def test_board_to_fen(self):
        board = generate_start_board()
        self.assertEqual(to_fen(board), '3prbnk/4ppqn/5ppb/P5pr/RP5p/BPP5/NQPP4/KNBRP3')

# run tests if this file is run directly
if __name__ == '__main__':
    unittest.main(exit=False)