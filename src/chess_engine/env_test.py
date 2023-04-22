
import random
import unittest

import numpy as np
from . import DiagonalChess, action



class DiagonalChessTests(unittest.TestCase):
    def test_random_moves(self):
        # random moves shall not trigger any errors
        env = DiagonalChess()
        env.reset()
        for _ in range(100):
            random_action = random.randrange(4096)
            action, _, done = env.step(random_action)
            
            self.assertEqual(action.shape, (6, 8, 8))
            self.assertEqual(action.dtype, np.int8)

            if done:
                env.reset()
        
        self.assertTrue(True)
            
    def test_move_to_action(self):
        self.assertEqual(action('a1a1'), 0+0*8+0*64+0*512)
        self.assertEqual(action('a1a2'), 0+0*8+0*64+1*512)
        self.assertEqual(action('a1a3'), 0+0*8+0*64+2*512)
        self.assertEqual(action('a1a4'), 0+0*8+0*64+3*512)

        self.assertEqual(action('a1b1'), 0+0*8+1*64+0*512)
        self.assertEqual(action('a1b2'), 0+0*8+1*64+1*512)
        self.assertEqual(action('a1b3'), 0+0*8+1*64+2*512)

        self.assertEqual(action('a1b4'), 0+0*8+1*64+3*512)
        self.assertEqual(action('a1c1'), 0+0*8+2*64+0*512)
        self.assertEqual(action('a1c2'), 0+0*8+2*64+1*512)

    

