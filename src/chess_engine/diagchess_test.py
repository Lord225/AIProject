import unittest

import numpy as np

from .diagchess import board_to_observation, generate_start_board, pawn_legal_moves, rook_legal_moves, bishop_legal_moves, piece, piece_to_fen, to_fen

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
        self.assertTrue(np.array_equal(movesp, legal_movesp))
        self.assertTrue(np.array_equal(movesP, legal_movesP))


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
        self.assertTrue(np.array_equal(moves, legal_moves))

        board[5,5] = piece("PAWN")
        moves = pawn_legal_moves(board,5,5)
        legal_moves = np.zeros((8, 8), dtype=np.int8)
        legal_moves[6,6] = piece("PAWN")
        legal_moves[4,4] = piece("PAWN")
        legal_moves[6,4] = piece("PAWN")
        # print(moves)
        # print(legal_moves)
        self.assertTrue(np.array_equal(moves, legal_moves))

        board[5,4] = 0
        board[6,5] = 0
        moves = pawn_legal_moves(board,5,5)
        legal_moves[5,4] = piece("PAWN")
        legal_moves[6,5] = piece("PAWN")
        self.assertTrue(np.array_equal(moves, legal_moves))
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
        self.assertTrue(np.array_equal(moves, legal_moves))

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
        self.assertTrue(np.array_equal(moves, legal_moves))


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
        self.assertTrue(np.array_equal(moves, legal_moves))

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
        self.assertTrue(np.array_equal(moves, legal_moves))


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
        self.assertTrue(np.array_equal(moves, legal_moves))


class TestObservation(unittest.TestCase):
    def test_board_to_observation(self):
        board = np.zeros((8, 8), dtype=np.int8)
        board[5,5] = piece("pawn")
        board[2,2] = piece("PAWN")

        observation = board_to_observation(board)
        legal_observation = np.zeros((6, 8, 8), dtype=np.int8)
        legal_observation[0, 5, 5] = 1
        legal_observation[0, 2, 2] = -1

        self.assertTrue(np.array_equal(observation, legal_observation))
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

        self.assertTrue(np.array_equal(expected, observation))



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
