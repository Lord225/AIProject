import unittest

import numpy as np

from .diagchess import *

def array_equal_print(arr1: np.ndarray, arr2: np.ndarray) -> bool:
    if np.array_equal(arr1, arr2):
        return True
    else:
        print("Arrays not equal!")
        print("first array:")
        print(arr1)
        print("second array:")
        print(arr2)
        return False


class TestLegalMoves(unittest.TestCase):
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
        self.assertTrue(array_equal_print(movesp, legal_movesp))
        self.assertTrue(array_equal_print(movesP, legal_movesP))


        board = np.zeros((8, 8), dtype=np.int8)
        board[5,5] = piece("pawn")

        board[4,5] = piece("rook")
        board[4,4] = piece("rook")
        board[4,6] = piece("rook")
        board[5,6] = piece("rook")
        board[5,4] = piece("rook")
        board[6,4] = piece("rook")
        board[6,5] = piece("rook")
        board[6,6] = piece("rook")
        moves = pawn_legal_moves(board,5,5)
        legal_moves = np.zeros((8, 8), dtype=np.int8)
        self.assertTrue(array_equal_print(moves, legal_moves))

        board[5,5] = piece("PAWN")
        moves = pawn_legal_moves(board,5,5)
        legal_moves = np.zeros((8, 8), dtype=np.int8)
        legal_moves[6,6] = piece("PAWN")
        legal_moves[4,4] = piece("PAWN")
        legal_moves[6,4] = piece("PAWN")
        # print(moves)
        # print(legal_moves)
        self.assertTrue(array_equal_print(moves, legal_moves))

        board[5,4] = 0
        board[6,5] = 0
        moves = pawn_legal_moves(board,5,5)
        legal_moves[5,4] = piece("PAWN")
        legal_moves[6,5] = piece("PAWN")
        self.assertTrue(array_equal_print(moves, legal_moves))
        # print(moves)
        # print(legal_moves)

        board = np.array([  [0, 0, 0, 0, 0, 0, 0 , 0],
                            [0, 0, 0, 0, 0, 0, 0 , 0],
                            [0, 0, 0, 0, 0, 0, 0 , 0],
                            [0, 0, 0, 0, 0, 0, 0 , 0],
                            [0, 0, 0, 0, 0, 0, 0 , 0],
                            [0, 0, -1, 0, 0, 0, 0 , 0],
                            [0, 0, 0, 0, 0, 0, 0 , 0],
                            [0, 0, 0, 0, 0, 0, 0 , 0],]).astype(np.int8)

        legal_moves = np.array(  [  [0, 0, 0, 0, 0, 0, 0 , 0],
                                    [0, 0, 0, 0, 0, 0, 0 , 0],
                                    [0, 0, 0, 0, 0, 0, 0 , 0],
                                    [0, 0, -1, 0, 0, 0, 0 , 0],
                                    [0, 0, -1, 0, 0, 0, 0 , 0],
                                    [0, 0, 0, -1, -1, 0, 0 , 0],
                                    [0, 0, 0, 0, 0, 0, 0 , 0],
                                    [0, 0, 0, 0, 0, 0, 0 , 0],]).astype(np.int8)

        moves = pawn_legal_moves(board,2,5)
        # print(moves)
        # print(legal_moves)
        self.assertTrue(array_equal_print(moves, legal_moves))

    def test_rook_legal_moves(self):
        board = np.zeros((8, 8), dtype=np.int8)
        board[5,5] = piece("ROOK")

        legal_moves = np.array([[0, 0, 0, 0, 0, 2, 0, 0],
                                [0, 0, 0, 0, 0, 2, 0, 0],
                                [0, 0, 0, 0, 0, 2, 0, 0],
                                [0, 0, 0, 0, 0, 2, 0, 0],
                                [0, 0, 0, 0, 0, 2, 0, 0],
                                [2, 2, 2, 2, 2, 0, 2, 2],
                                [0, 0, 0, 0, 0, 2, 0, 0],
                                [0, 0, 0, 0, 0, 2, 0, 0],]).astype(np.int8)
        moves = rook_legal_moves(board,5,5)

        # print(moves)
        # print(legal_moves)
        self.assertTrue(array_equal_print(moves, legal_moves))


        legal_moves = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 2, 0, 0],
                                [0, 0, 0, 0, 0, 2, 0, 0],
                                [0, 0, 0, 0, 0, 2, 0, 0],
                                [0, 0, 0, 0, 0, 2, 0, 0],
                                [0, 0, 2, 2, 2, 0, 2, 0],
                                [0, 0, 0, 0, 0, 2, 0, 0],
                                [0, 0, 0, 0, 0, 2, 0, 0],]).astype(np.int8)
        board[5,1] = piece("ROOK")
        board[1,5] = piece("rook")
        board[5,6] = piece("pawn")
        moves = rook_legal_moves(board,5,5)
        # print(moves)
        # print(legal_moves)
        self.assertTrue(array_equal_print(moves, legal_moves))

    def test_bishop_legal_moves(self):
        board = np.zeros((8, 8), dtype=np.int8)
        board[5,5] = piece("BISHOP")

        legal_moves = np.array([[4, 0, 0, 0, 0, 0, 0, 0],
                                [0, 4, 0, 0, 0, 0, 0, 0],
                                [0, 0, 4, 0, 0, 0, 0, 0],
                                [0, 0, 0, 4, 0, 0, 0, 4],
                                [0, 0, 0, 0, 4, 0, 4, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 4, 0, 4, 0],
                                [0, 0, 0, 4, 0, 0, 0, 4],]).astype(np.int8)
        moves = bishop_legal_moves(board,5,5)

        # print(moves)
        # print(legal_moves)
        self.assertTrue(array_equal_print(moves, legal_moves))


        legal_moves = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 4, 0, 0, 0, 0, 0],
                                [0, 0, 0, 4, 0, 0, 0, 0],
                                [0, 0, 0, 0, 4, 0, 4, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 4, 0, 4, 0],
                                [0, 0, 0, 0, 0, 0, 0, 4],]).astype(np.int8)
        board[1,1] = piece("ROOK")
        board[6,4] = piece("bishop")
        board[4,6] = piece("pawn")
        moves = bishop_legal_moves(board,5,5)
        print(moves)
        print(legal_moves)
        self.assertTrue(array_equal_print(moves, legal_moves))
        
    def test_queen_legal_moves(self):
        board = np.zeros((8, 8), dtype=np.int8)
        board[5,5] = piece("queen")

        legal_moves = np.array([[-5, 0 , 0 , 0 , 0 ,-5 , 0 , 0],
                                [0 ,-5 , 0 , 0 , 0 ,-5 , 0 , 0],
                                [0 , 0 ,-5 , 0 , 0 ,-5 , 0 , 0],
                                [0 , 0 , 0 ,-5 , 0 ,-5 , 0 ,-5],
                                [0 , 0 , 0 , 0 ,-5 ,-5 ,-5 , 0],
                                [-5,-5 ,-5 ,-5 ,-5 , 0 ,-5 ,-5],
                                [0 , 0 , 0 , 0 ,-5 ,-5 ,-5 , 0],
                                [0 , 0 , 0 ,-5 , 0 ,-5 , 0 ,-5],]).astype(np.int8)
        moves = queen_legal_moves(board,5,5)

        # print(moves)
        # print(legal_moves)
        self.assertTrue(array_equal_print(moves, legal_moves))


        legal_moves = np.array([[0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 ,-5 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 ,-5 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 ,-5 , 0 ,-5 , 0 , 0],
                                [0 , 0 , 0 , 0 ,-5 ,-5 , 0 , 0],
                                [-5,-5 ,-5 ,-5 ,-5 , 0 ,-5 , 0],
                                [0 , 0 , 0 , 0 , 0 ,-5 ,-5 , 0],
                                [0 , 0 , 0 , 0 , 0 ,-5 , 0 ,-5],]).astype(np.int8)
        board[1,1] = piece("ROOK")
        board[6,4] = piece("bishop")
        board[4,6] = piece("pawn")
        board[2,5] = piece("knight")
        board[7,5] = piece("KNIGHT")
        board[5,6] = piece("KNIGHT")
        moves = queen_legal_moves(board,5,5)
        print(moves)
        print(legal_moves)
        self.assertTrue(array_equal_print(moves, legal_moves))

    def test_knight_legal_moves(self):
        board = np.zeros((8, 8), dtype=np.int8)
        board[5,5] = piece("knight")

        legal_moves = np.array([[0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 , 0 ,-3 , 0 ,-3 , 0],
                                [0 , 0 , 0 ,-3 , 0 , 0 , 0 ,-3],
                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 ,-3 , 0 , 0 , 0 ,-3],
                                [0 , 0 , 0 , 0 ,-3 , 0 ,-3 , 0],]).astype(np.int8)
        moves = knight_legal_moves(board,5,5)

        self.assertTrue(array_equal_print(moves, legal_moves))

        board = np.zeros((8, 8), dtype=np.int8)
        board[1,1] = piece("knight")

        legal_moves = np.array([[0 , 0 , 0 ,-3 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 ,-3 , 0 , 0 , 0 , 0],
                                [-3, 0 ,-3 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],]).astype(np.int8)
        moves = knight_legal_moves(board,1,1)

        # print(moves)
        # print(legal_moves)
        self.assertTrue(array_equal_print(moves, legal_moves))

        board = np.zeros((8, 8), dtype=np.int8)
        board[5,5] = piece("knight")
        legal_moves = np.array([[0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 ,-3 , 0],
                                [0 , 0 , 0 ,-3 , 0 , 0 , 0 ,-3],
                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 ,-3 , 0 , 0 , 0 ,-3],
                                [0 , 0 , 0 , 0 , 0 , 0 ,-3 , 0],]).astype(np.int8)
        board[4,3] = piece("ROOK")
        board[3,4] = piece("bishop")
        board[7,4] = piece("pawn")
        moves = knight_legal_moves(board,5,5)
        # print(moves)
        # print(legal_moves)
        self.assertTrue(array_equal_print(moves, legal_moves))

    def test_king_legal_moves(self):
        board = np.zeros((8, 8), dtype=np.int8)
        board[5,5] = piece("KING")


        legal_moves = np.array([[0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 , 0 , 6 , 6 , 6 , 0],
                                [0 , 0 , 0 , 0 , 6 , 0 , 6 , 0],
                                [0 , 0 , 0 , 0 , 6 , 6 , 6 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],]).astype(np.int8)
        moves = king_legal_moves(board,5,5)

        # print(moves)
        # print(legal_moves)
        self.assertTrue(array_equal_print(moves, legal_moves))
        board[4,3] = piece("king")


        legal_moves = np.array([[0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 , 6 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 , 6 , 0],
                                [0 , 0 , 0 , 0 , 6 , 6 , 6 , 0],
                                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],]).astype(np.int8)
        board[4,5] = piece("ROOK")
        board[5,6] = piece("rook")

        moves = king_legal_moves(board,5,5)
        print(moves)
        print(legal_moves)
        self.assertTrue(array_equal_print(moves, legal_moves))


class TestObservation(unittest.TestCase):
    def test_board_to_observation(self):
        board = np.zeros((8, 8), dtype=np.int8)
        board[5,5] = piece("pawn")
        board[2,2] = piece("PAWN")

        observation = board_to_observation(board)
        legal_observation = np.zeros((6, 8, 8), dtype=np.int8)
        legal_observation[0, 5, 5] = 1
        legal_observation[0, 2, 2] = -1

        self.assertTrue(array_equal_print(observation, legal_observation))
    def test_start_board_as_observation(self):
        board = generate_start_board()

        observation = board_to_observation(board)

        expected = np.array(
            [[
            [ 0 , 0,  0, -1,  0,  0,  0,  0],
            [ 0  ,0 , 0 , 0 ,-1 ,-1 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 ,-1 ,-1 , 0],
            [ 1  ,0 , 0 , 0 , 0 , 0 ,-1 , 0],
            [ 0  ,1 , 0 , 0 , 0 , 0 , 0 ,-1],
            [ 0  ,1 , 1 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 1 , 1 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 1 , 0 , 0 , 0]],
            [
            [ 0  ,0 ,  0,  0, -1,  0,  0,  0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 ,-1],
            [ 1  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 1 , 0 , 0 , 0 , 0]],
            [
            [ 0  ,0,  0,  0,  0,  0, -1,  0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 ,-1],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 1  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,1 , 0 , 0 , 0 , 0 , 0 , 0]],
            [
            [ 0 , 0 ,  0,  0,  0, -1,  0,  0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 ,-1],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 1  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 1 , 0 , 0 , 0 , 0 , 0]],
            [
            [ 0  ,0 ,  0,  0,  0,  0,  0,  0],
            [ 0  ,0 , 0 , 0 , 0 , 0 ,-1 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,1 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0]],
            [
            [ 0  ,0 , 0,  0,  0,  0,  0, -1],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 0  ,0 , 0 , 0 , 0 , 0 , 0 , 0],
            [ 1  ,0 , 0 , 0 , 0 , 0 , 0 , 0]]]
        )

        self.assertTrue(array_equal_print(expected, observation))



class TestTransforms(unittest.TestCase):
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


class TestMoveChoosing(unittest.TestCase):
    """
    Board cheetsheet:
      0 1 2 3 4 5 6 7
    0       p r b n k 
    1         p p q n 
    2           p p b 
    3 P           p r 
    4 R P           p 
    5 B P P           
    6 N Q P P         
    7 K N B R P       
    """
    def test_legal_moves(self):
        board = generate_start_board()
        
        # move a pawn as white
        move = generate_move(board, 0, 3, 1, 3, False)
        # moves, reward for legal move
        self.assertEqual(move, ((0, 3, 1, 3), LEGAL_MOVE_REWARD))

        # move a pawn as black
        move = generate_move(board, 3, 0, 3, 2, True)
        self.assertEqual(move, ((3, 0, 3, 2), LEGAL_MOVE_REWARD))

    def test_legal_move_as_other_color(self):
        board = generate_start_board()
        # move wite pawn as black
        _, reward = generate_move(board, 1, 4, 2, 4, True)
        self.assertEqual(reward, WRONG_PIECE_COLOR_PENALTY)
    def test_illegal_moves(self):
        board = generate_start_board()

        # move a pawn as white onto illegal square
        _, reward = generate_move(board, 2, 5, 0, 0, False)
        self.assertEqual(reward, ILLEGAL_MOVE_PENALTY_1)

        # try moving king
        _, reward = generate_move(board, 0, 7, 0, 6, False)
        self.assertEqual(reward, ILLEGAL_MOVE_PENALTY_2)
