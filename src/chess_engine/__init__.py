from typing import Tuple
import unittest
import numpy as np
import numba as nb
import chess
import chess.svg

@nb.njit('int8(types.unicode_type)',cache=True)
def piece(name: str) -> int:
    pieceS = {
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

    return pieceS[name]

@nb.njit(cache=True)
def piece_to_fen(piece: int) -> str:
    piece_map = {
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

    return piece_map[piece]

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
                fen += piece_to_fen(piece)
        if empty > 0:
            fen += str(empty)
        fen += '/'
    fen = fen[:-1]
    return fen

def inbounds(x, y):
    return 0 <= x < 8 and 0 <= y < 8

def is_starting_position(x, y, piece):
    initial_board = generate_start_board()
    return initial_board[y, x] == piece


def pawn_legal_moves(board: np.ndarray, x: int, y: int):
    moves = np.zeros((8, 8), dtype=np.int8)
    piece = board[y, x]
    is_init_pos = is_starting_position(x, y, piece)
    
    # Determine the direction the pawn is moving based on its color
    if piece > 0:  # white pawn
        direction = 1
    else:  # black pawn
        direction = -1
    
    # Check if the pawn can move one square forward
    if inbounds(y+direction, x) and board[y+direction, x] == 0:
        moves[y+direction, x] = piece
        
        # Check if the pawn can move two squares forward from its starting position
        if inbounds(y+2*direction, x) and is_init_pos and board[y+2*direction, x] == 0:
            moves[y+2*direction, x] = piece
    # Check if the pawn can move one square on the side
    if inbounds(y, x - direction) and board[y, x - direction] == 0:
        moves[y, x - direction] = piece
        
        # Check if the pawn can move two squares forward from its starting position
        if inbounds(y, x - 2*direction)  and is_init_pos and board[y, x - 2*direction] == 0:
            moves[y, x - 2*direction] = piece
    
    # Check if the pawn can capture diagonally forward
    if inbounds(y+direction, x-direction) and board[y+direction, x-direction] * piece < 0:
        moves[y+direction, x-direction] = piece
        
    # Check if the pawn can capture diagonally to its left
    if inbounds(y-direction, x-direction) and board[y-direction, x-direction] * piece < 0:
        moves[y-direction, x-direction] = piece
    # Check if the pawn can capture diagonally to its right
    if inbounds(y+direction, x+direction) and board[y+direction, x+direction] * piece < 0:
        moves[y-direction, x-direction] = piece
    
    return moves


def legal_moves(board: np.ndarray, x,y):
    piece_value = board[y, x]
    
    if abs(piece_value) == piece('pawn'):
        return pawn_legal_moves(board, x, y)
    elif abs(piece_value) == piece('rook'):
        return rook_legal_moves(board, x, y)
    elif abs(piece_value) == piece('knight'):
        return knight_legal_moves(board, x, y)
    elif abs(piece_value) == piece('bishop'):
        return bishop_legal_moves(board, x, y)
    elif abs(piece_value) == piece('queen'):
        return queen_legal_moves(board, x, y)
    elif abs(piece_value) == piece('king'):
        return king_legal_moves(board, x, y)

@nb.njit('int8[:,:]()',cache=True)
def generate_start_board() -> np.ndarray:
    board = np.zeros((8, 8), dtype=np.int8)

    board[0, :] = np.array([0             , 0             , 0             , piece('PAWN'), piece('ROOK'), piece('BISHOP'), piece('KNIGHT'), piece('KING')  ], dtype=np.int8)
    board[1, :] = np.array([0             , 0             , 0             , 0           , piece('PAWN'), piece('PAWN')  , piece('QUEEN') , piece('KNIGHT')], dtype=np.int8)
    board[2, :] = np.array([0             , 0             , 0             , 0           , 0           , piece('PAWN')  , piece('PAWN')  , piece('BISHOP')], dtype=np.int8)
    board[3, :] = np.array([piece('pawn')  , 0             , 0             , 0           , 0           , 0             , piece('PAWN')  , piece('ROOK')  ], dtype=np.int8)
    board[4, :] = np.array([piece('rook')  , piece('pawn')  , 0             , 0           , 0           , 0             , 0             , piece('PAWN')  ], dtype=np.int8)
    board[5, :] = np.array([piece('bishop'), piece('pawn')  , piece('pawn')  , 0           , 0           , 0             , 0             , 0             ], dtype=np.int8)
    board[6, :] = np.array([piece('knight'), piece('queen') , piece('pawn')  , piece('pawn'), 0           , 0             , 0             , 0             ], dtype=np.int8)
    board[7, :] = np.array([piece('king')  , piece('knight'), piece('bishop'), piece('rook'), piece('pawn'), 0             , 0             , 0             ], dtype=np.int8)
    
    return board


def fen_to_svg(fen: str) -> str:
    return chess.svg.board(chess.Board(fen), size=500)

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

        return fen_to_svg(to_fen(self.board))
    
    def __str__(self):
        output = ''
        for row in self.board:
            output += f"{' '.join([piece_to_fen(piece) for piece in row])} \n"
        return output
    
    def __repr__(self):
        return to_fen(self.board)


class TestDiagonalChess(unittest.TestCase):
    def test_piece_to_unit(self):
        self.assertEqual(piece_to_fen(1), 'p')
        self.assertEqual(piece_to_fen(-1), 'P')
        self.assertEqual(piece_to_fen(2), 'r')
        self.assertEqual(piece_to_fen(-2), 'R')
        self.assertEqual(piece_to_fen(3), 'n')
        self.assertEqual(piece_to_fen(-3), 'N')
        self.assertEqual(piece_to_fen(4), 'b')
        self.assertEqual(piece_to_fen(-4), 'B')
        self.assertEqual(piece_to_fen(5), 'q')
        self.assertEqual(piece_to_fen(-5), 'Q')
        self.assertEqual(piece_to_fen(6), 'k')
        self.assertEqual(piece_to_fen(-6), 'K')
    
    def test_board_to_fen(self):
        board = generate_start_board()
        self.assertEqual(to_fen(board), '3prbnk/4ppqn/5ppb/P5pr/RP5p/BPP5/NQPP4/KNBRP3')
    def test_pawn_legal_moves(self):
        board = np.zeros((8, 8), dtype=np.int8)
        board[5,5] = piece("pawn")
        board[2,2] = piece("PAWN")

        movesp = pawn_legal_moves(board,5,5)
        legal_movesp = np.zeros((8, 8), dtype=np.int8)
        legal_movesp[4,5] = piece("pawn")
        legal_movesp[5,6] = piece("pawn")

        movesP = pawn_legal_moves(board,2,2)
        legal_movesP = np.zeros((8, 8), dtype=np.int8)
        legal_movesP[2,1] = piece("PAWN")
        legal_movesP[3,2] = piece("PAWN")
        self.assertTrue(np.array_equal(movesp, legal_movesp))
        self.assertTrue(np.array_equal(movesP, legal_movesP))

if __name__ == '__main__':
    board = generate_start_board()
    board = np.zeros((8, 8), dtype=np.int8)
    # board[5,5] = piece("pawn")
    board[2,2] = piece("PAWN")
    # moves = pawn_legal_moves(board,5,5)
    moves = pawn_legal_moves(board,2,2)

    print(moves)
