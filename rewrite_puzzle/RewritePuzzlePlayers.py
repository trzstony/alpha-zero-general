"""
Testing player for the Rewrite Puzzle game.
Useful for basic testing of game logic.
"""

import numpy as np


class RandomPlayer():
    """
    Random player for testing purposes.
    Selects a random valid move at each turn.
    Useful for:
    - Basic smoke testing (does the game run without crashing?)
    - Testing that valid moves work correctly
    - Quick sanity checks
    """
    def __init__(self, game):
        self.game = game

    def play(self, board):
        """Play a random valid move."""
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a

