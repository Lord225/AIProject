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

@nb.njit(cache=True)
def inbounds(x: int, y: int):
    return 0 <= x < 8 and 0 <= y < 8

@nb.njit(cache=True)
def is_starting_position(x: int, y: int, piece: int):
    initial_board = generate_start_board()
    return initial_board[y, x] == piece

@nb.njit('int8[:,:](int8[:,:], int32, int32)', cache=True)
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
        return np.zeros((8, 8), dtype=np.int8)
        return rook_legal_moves(board, x, y)
    elif abs(piece_value) == piece('knight'):
        return np.zeros((8, 8), dtype=np.int8)
        return knight_legal_moves(board, x, y)
    elif abs(piece_value) == piece('bishop'):
        return np.zeros((8, 8), dtype=np.int8)
        return bishop_legal_moves(board, x, y)
    elif abs(piece_value) == piece('queen'):
        return np.zeros((8, 8), dtype=np.int8)
        return queen_legal_moves(board, x, y)
    elif abs(piece_value) == piece('king'):
        return np.zeros((8, 8), dtype=np.int8)
        return king_legal_moves(board, x, y)

    return np.zeros((8, 8), dtype=np.int8)

@nb.njit('int8[:,:,:](int8[:,:])', cache=True)
def board_to_observation(board: np.ndarray) -> np.ndarray:
    observation = np.zeros((6, 8, 8), dtype=np.int8)

    observation[0, :, :] = (board == piece('pawn')).astype(np.int8) - (board == piece('PAWN')).astype(np.int8)
    observation[1, :, :] = (board == piece('rook')).astype(np.int8) - (board == piece('ROOK')).astype(np.int8)
    observation[2, :, :] = (board == piece('knight')).astype(np.int8) - (board == piece('KNIGHT')).astype(np.int8)
    observation[3, :, :] = (board == piece('bishop')).astype(np.int8) - (board == piece('BISHOP')).astype(np.int8)
    observation[4, :, :] = (board == piece('queen')).astype(np.int8) - (board == piece('QUEEN')).astype(np.int8)
    observation[5, :, :] = (board == piece('king')).astype(np.int8) - (board == piece('KING')).astype(np.int8)

    return observation


def fen_to_svg(fen: str) -> str:
    return chess.svg.board(chess.Board(fen), size=500)


if __name__ == '__main__':
    board = generate_start_board()
    print(board_to_observation(board))
    board = np.zeros((8, 8), dtype=np.int8)
    # board[5,5] = piece("pawn")
    board[2,2] = piece("PAWN")
    # moves = pawn_legal_moves(board,5,5)
    moves = pawn_legal_moves(board,2,2)

    print(moves)