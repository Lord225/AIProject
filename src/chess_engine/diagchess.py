from typing import Optional, Tuple
import numpy as np
import numba as nb
import chess
import chess.svg

WRONG_PIECE_COLOR_PENALTY = -0.1
ILLEGAL_MOVE_PENALTY_1 = -0.1
ILLEGAL_MOVE_PENALTY_2 = -0.1
LEGAL_MOVE_REWARD = 1

PAWN_CAPTURE_REWARD = 5
ROOK_CAPTURE_REWARD = 5
KNIGHT_CAPTURE_REWARD = 5
BISHOP_CAPTURE_REWARD = 5
QUEEN_CAPTURE_REWARD = 5
KING_CAPTURE_REWARD = 5

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
        moves[y+direction, x+direction] = piece
    
    return moves

@nb.njit('int8[:,:](int8[:,:], int32, int32)', cache=True)
def rook_legal_moves(board: np.ndarray, x: int, y: int):
    moves = np.zeros((8, 8), dtype=np.int8)
    piece = board[y, x]
    
    # Check valid moves along the x-axis
    for i in range(x + 1, 8):
        if board[y, i] == 0:
            moves[y, i] = piece
        elif board[y, i] * piece < 0:
            moves[y, i] = piece
            break
        else:
            break

    for i in range(x - 1, -1, -1):
        if board[y, i] == 0:
            moves[y, i] = piece
        elif board[y, i] * piece < 0:
            moves[y, i] = piece
            break
        else:
            break

    # Check valid moves along the y-axis
    for i in range(y + 1, 8):
        if board[i, x] == 0:
            moves[i, x] = piece
        elif board[i, x] * piece < 0:
            moves[i, x] = piece
            break
        else:
            break

    for i in range(y - 1, -1, -1):
        if board[i, x] == 0:
            moves[i, x] = piece
        elif board[i, x] * piece < 0:
            moves[i, x] = piece
            break
        else:
            break

    return moves

@nb.njit('int8[:,:](int8[:,:], int32, int32)', cache=True)
def bishop_legal_moves(board: np.ndarray, x: int, y: int):
    moves = np.zeros((8, 8), dtype=np.int8)
    piece = board[y, x]

    # Check valid moves along the top-left to bottom-right diagonal
    i, j = x + 1, y + 1
    while i < 8 and j < 8:
        if board[j, i] == 0:
            moves[j, i] = piece
        elif board[j, i] * piece < 0:
            moves[j, i] = piece
            break
        else:
            break
        i += 1
        j += 1

    i, j = x - 1, y - 1
    while i >= 0 and j >= 0:
        if board[j, i] == 0:
            moves[j, i] = piece
        elif board[j, i] * piece < 0:
            moves[j, i] = piece
            break
        else:
            break
        i -= 1
        j -= 1

    # Check valid moves along the top-right to bottom-left diagonal
    i, j = x + 1, y - 1
    while i < 8 and j >= 0:
        if board[j, i] == 0:
            moves[j, i] = piece
        elif board[j, i] * piece < 0:
            moves[j, i] = piece
            break
        else:
            break
        i += 1
        j -= 1

    i, j = x - 1, y + 1
    while i >= 0 and j < 8:
        if board[j, i] == 0:
            moves[j, i] = piece
        elif board[j, i] * piece < 0:
            moves[j, i] = piece
            break
        else:
            break
        i -= 1
        j += 1

    return moves

@nb.njit('int8[:,:](int8[:,:], int32, int32)', cache=True)
def queen_legal_moves(board: np.ndarray, x: int, y: int):
    # piece = board[y, x]
    bishop_moves = bishop_legal_moves(board,x,y)
    rook_moves = rook_legal_moves(board,x,y)
    moves = np.add(bishop_moves, rook_moves)
    # return np.where(moves != 0, piece, moves)
    return moves

@nb.njit('int8[:,:](int8[:,:], int32, int32)', cache=True)
def knight_legal_moves(board: np.ndarray, x: int, y: int):
    moves = np.zeros((8, 8), dtype=np.int8)
    piece = board[y, x]
    
    # Possible knight moves
    knight_moves = [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)]
    
    for move in knight_moves:
        # Determine potential square to move to
        new_x = x + move[0]
        new_y = y + move[1]
        
        # Check if the move is within the bounds of the board and is either empty or contains an opponent piece
        if inbounds(new_y, new_x) and (board[new_y, new_x] == 0 or board[new_y, new_x] * piece < 0):
            moves[new_y, new_x] = piece
            
    return moves

@nb.njit('int8[:,:](int8[:,:], int32, int32)', cache=True)
def king_legal_moves(board: np.ndarray, x: int, y: int):
    moves = np.zeros((8, 8), dtype=np.int8)
    piece = board[y, x]
    
    # Possible king moves
    king_moves = [(1,1), (1,-1), (-1,1), (-1,-1), (1,0), (-1,0), (0,1), (0,-1)]
    
    for move in king_moves:
        # Determine potential square to move to
        new_x = x + move[0]
        new_y = y + move[1]
        
        # Check if the move is within the bounds of the board and is either empty or contains an opponent piece
        if inbounds(new_y, new_x) and (board[new_y, new_x] == 0 or board[new_y, new_x] * piece < 0):
            
            # Check if the king is not moving adjacent to an opponent king
            king_positions = np.where(board == -piece)
            for kx, ky in zip(king_positions[1], king_positions[0]):
                if abs(new_x - kx) <= 1 and abs(new_y - ky) <= 1:
                    break
            else:
                moves[new_y, new_x] = piece
            
    return moves

@nb.njit('int8[:,:](int8[:,:], int32, int32)', cache=True)
def legal_moves(board: np.ndarray, x, y):
    piece_value = board[y, x]
    
    if abs(piece_value) == piece('PAWN'):
        return pawn_legal_moves(board, x, y)
    elif abs(piece_value) == piece('ROOK'):
        return rook_legal_moves(board, x, y)
    elif abs(piece_value) == piece('KNIGHT'):
        return knight_legal_moves(board, x, y)
    elif abs(piece_value) == piece('BISHOP'):
        return bishop_legal_moves(board, x, y)
    elif abs(piece_value) == piece('QUEEN'):
        return queen_legal_moves(board, x, y)
    elif abs(piece_value) == piece('king'):
        return king_legal_moves(board, x, y)

    return np.zeros((8, 8), dtype=np.int8)

@nb.njit('int8[:,:](int8[:,:], boolean)', cache=True)
def all_legal_moves(board: np.ndarray, isBlack: bool) -> np.ndarray:
    moves = np.zeros((8, 8), dtype=np.int8)
    xs, ys = np.where((board < 0) != isBlack)
    for x, y in zip(xs, ys):
        moves += legal_moves(board, y, x)
    return moves

@nb.njit('float32[:,:,:](int8[:,:])', cache=True)
def board_to_observation(board: np.ndarray) -> np.ndarray:
    observation = np.zeros((8, 8, 8), dtype=np.float32)

    observation[:, :, 0] = (board == piece('pawn')).astype(np.int8) - (board == piece('PAWN')).astype(np.int8)
    observation[:, :, 1] = (board == piece('rook')).astype(np.int8) - (board == piece('ROOK')).astype(np.int8)
    observation[:, :, 2] = (board == piece('knight')).astype(np.int8) - (board == piece('KNIGHT')).astype(np.int8)
    observation[:, :, 3] = (board == piece('bishop')).astype(np.int8) - (board == piece('BISHOP')).astype(np.int8)
    observation[:, :, 4] = (board == piece('queen')).astype(np.int8) - (board == piece('QUEEN')).astype(np.int8)
    observation[:, :, 5] = (board == piece('king')).astype(np.int8) - (board == piece('KING')).astype(np.int8)

    observation[:, :, 6] = all_legal_moves(board, True)
    observation[:, :, 7] = all_legal_moves(board, False)

    return observation

@nb.njit('float32[:,:,:,:](int8[:,:,:])', cache=True)
def board_to_observation_batch(board: np.ndarray) -> np.ndarray:
    output = np.zeros((len(board), 8, 8, 8), dtype=np.float32)
    for i in range(len(board)):
        output[i] = board_to_observation(board[i])
    return output


@nb.njit(cache=True)
def random_legal_move(board: np.ndarray, isBlack: bool) -> Optional[Tuple[int, int, int, int]]:
    # choose random piece
    pieces = np.argwhere((board > 0) != isBlack)

    # if no pieces, return None
    if len(pieces) == 0:
        return None
    
    # random permutation of pieces
    np.random.shuffle(pieces)

    for x1, y1 in pieces:
        legal = legal_moves(board, y1, x1)

        # choose random legal move
        legal = np.argwhere(legal != 0)

        if len(legal) == 0:
            continue
        
        # choose random legal move
        x2, y2 = legal[np.random.randint(0, len(legal))]

        return (x1, y1, x2, y2)
    return None

@nb.njit(cache=True)
def generate_move(board: np.ndarray, x1: int, y1: int, x2: int, y2: int, isBlack: bool) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    """
    generates legal move and penalty from any illegal move, return None if no legal moves are possible
    """
    # check if move is legal
    piece = board[y1, x1]

    # check if piece is correct color
    if (piece > 0) != isBlack:
        move = random_legal_move(board, isBlack)
        if move is None:
            return None, 0 # no legal moves, game over
        else:
            return move, WRONG_PIECE_COLOR_PENALTY # wrong piece color
        
    # check what are the legal moves
    legal = legal_moves(board, x1, y1)

    if legal[y2, x2] != 0:
        # legal move
        return (x1, y1, x2, y2), LEGAL_MOVE_REWARD
    else:
        # choose random legal move
        legal = np.argwhere(legal != 0)
        if len(legal) != 0:
            y2, x2 = legal[np.random.randint(0, len(legal))]
            return (x1, y1, x2, y2), ILLEGAL_MOVE_PENALTY_1 # legal pawn, illegal move
        else: 
            # no legal moves, try any move
            return random_legal_move(board, isBlack), ILLEGAL_MOVE_PENALTY_2 # no legal moves

@nb.njit(cache=True)
def capture_reward(captured_piece: int):
    if captured_piece == piece('PAWN'):
        return PAWN_CAPTURE_REWARD
    elif captured_piece == piece('ROOK'):
        return ROOK_CAPTURE_REWARD
    elif captured_piece == piece('KNIGHT'):
        return KNIGHT_CAPTURE_REWARD
    elif captured_piece == piece('BISHOP'):
        return BISHOP_CAPTURE_REWARD
    elif captured_piece == piece('QUEEN'):
        return QUEEN_CAPTURE_REWARD
    elif captured_piece == piece('KING'):
        return KING_CAPTURE_REWARD
    else:
        return 0

@nb.njit(cache=True)
def move_to_int(x1: int, y1: int, x2: int, y2: int) -> int:
    return (x1%8) * 8*8*8 + (y1%8) * 8*8 + (x2%8) * 8 + (y2%8)

@nb.njit(cache=True)
def int_action_to_move(action: int) -> Tuple[int, int, int, int]:
    x1 = (action // 8 // 8 // 8) % 8
    y1 = (action // 8 // 8) % 8
    x2 = (action // 8) % 8
    y2 = (action) % 8

    return x1, y1, x2, y2

@nb.njit('int32(int8[:,:], float32[:,:,:], boolean)',cache=True)
def array_action_to_move(board: np.ndarray, action: np.ndarray, isBlack: bool) -> int:
    # action is array 8x8x2, split into 8x8 and 8x8
    from_move = action[:, :, 0] 
    to_move = action[:, :, 1]

    # take all allay positions
    legal_from = (board < 0) != isBlack
    moves_from = from_move * legal_from  
    
    # get max position
    xf = np.argmax(moves_from)

    # get x y, unravel_index is not working with numpy
    x1 = xf % 8
    y1 = (xf // 8) % 8

    # check possible moves
    legal_to = np.abs(legal_moves(board, x1, y1))
    moves_to = to_move * legal_to
    
    # if no legal moves, choose random legal move
    if moves_to.sum() == 0:
        move = random_legal_move(board, isBlack)

        if move is None:
            return 0
        else:
            x1, y1, x2, y2 = move
            return move_to_int(x1, y1, x2, y2)
        
    
    # get max
    xt = np.argmax(moves_to)

    # get x y, unravel_index is not working with numpy
    x2 = xt % 8
    y2 = (xt // 8) % 8


    return move_to_int(x1, y1, x2, y2) # type: ignore

# vectorized version of array_action_to_move (takes action as array of Nx8x8x2)
@nb.njit('int32[:](int8[:,:,:], float32[:,:,:,:], boolean)',cache=True)
def array_action_to_move_vectorized(board: np.ndarray, action: np.ndarray, isBlack: bool) -> np.ndarray:
    output = np.zeros(action.shape[0], dtype=np.int32)
    for i in range(action.shape[0]):
        output[i] = array_action_to_move(board[i], action[i], isBlack)
    return output

# vectorized version of array_action_to_move (takes action as array of Nx8x8x2)
@nb.njit('int32[:](int8[:,:], float32[:,:,:,:], boolean)',cache=True)
def array_action_to_move_vectorized_one_board(board: np.ndarray, action: np.ndarray, isBlack: bool) -> np.ndarray:
    output = np.zeros(action.shape[0], dtype=np.int32)
    for i in range(action.shape[0]):
        output[i] = array_action_to_move(board, action[i], isBlack)
    return output
        
        

@nb.njit(cache=True)
def make_a_move(board: np.ndarray, x1: int, y1: int, x2: int, y2: int, isBlack: bool) -> Tuple[bool, float]:
    #print("chosed move", x1, y1, x2, y2)
    move, reward = generate_move(board, x1, y1, x2, y2, isBlack) # type: ignore
    if move is None:
        return True, 0
    else:
        x1, y1, x2, y2 = move
        #print("used move", x1, y1, x2, y2)
        # get piece
        piece = board[y1, x1]
        
        # get target piece
        target_piece = board[y2, x2]

        # get reward
        reward += capture_reward(target_piece)

        # move piece
        board[y2, x2] = piece

        # remove piece from old position
        board[y1, x1] = 0
    
    return False, reward

@nb.njit(cache=True)
def make_move_from_action(board: np.ndarray, action: int, isBlack: bool) -> Tuple[bool, float]:
    x1, y1, x2, y2 = int_action_to_move(action)
    return make_a_move(board, x1, y1, x2, y2, isBlack)

@nb.njit(cache=True)
def make_move_from_prob(board: np.ndarray, prob: np.ndarray, isBlack: bool) -> Tuple[bool, float]:
    action = array_action_to_move(board, prob, isBlack)
    if action is None:
        return True, 0
    
    return make_move_from_action(board, action, isBlack)

def fen_to_svg(fen: str) -> str:
    return chess.svg.board(chess.Board(fen), size=500)


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    board = generate_start_board()

    actions = np.random.rand(8, 8, 2)

    print(array_action_to_move(board, actions, True))

