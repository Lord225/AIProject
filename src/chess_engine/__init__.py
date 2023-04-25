from typing import Tuple
import numpy as np

from . import diagchess as internal

def action(move_str: str) -> int:
    x1ord = ord(move_str[0]) - ord("a")
    y1ord = 8 - (ord(move_str[1]) - ord("0"))
    x2ord = ord(move_str[2]) - ord("a")
    y2ord = 8 - (ord(move_str[3]) - ord("0"))

    print(x1ord, y1ord, x2ord, y2ord)

    return x1ord * 8 * 8 * 8 + y1ord * 8 * 8 + x2ord * 8 + y2ord


class DiagonalChess:
    def __init__(self):
        self.reset()


    def reset(self):
        """
        resets the board to the starting position
        """
        
        self.board = internal.generate_start_board()
        self.isBlack = False
    

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
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

        # make move
        done, reward = internal.make_move_from_action(self.board, action, self.isBlack)

        # switch player
        self.isBlack = not self.isBlack
        
        return internal.board_to_observation(self.board, self.isBlack), reward, done

    def step_human(self, move: str) -> Tuple[np.ndarray, float, bool]:
        return self.step(action(move))
    
    def step_cords(self, from_x: int, from_y: int, to_x: int, to_y: int) -> Tuple[np.ndarray, float, bool]:
        return self.step(from_x + from_y * 8 + to_x * 64 + to_y * 512)
    
    def step_prop(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        done, reward = internal.make_move_from_prob(self.board, action, self.isBlack)

        # switch player
        self.isBlack = not self.isBlack

        return internal.board_to_observation(self.board, self.isBlack), reward, done


    
    def render(self):
        """
        Should render the board using the python-chess library
        """

        return internal.fen_to_svg(internal.to_fen(self.board))

    def allowed_moves(self):
        return internal.all_legal_moves(self.board, self.isBlack)
    
    
    def __str__(self):
        output = ''
        for row in self.board:
            output += f"{' '.join([internal.piece_to_fen(piece) for piece in row])} \n"
        return output
    
    def __repr__(self):
        return internal.to_fen(self.board)
