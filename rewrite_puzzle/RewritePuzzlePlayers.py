"""
Testing player for the Rewrite Puzzle game (single-player).
Useful for basic testing of game logic.
"""

import numpy as np


class RandomPlayer():
    """
    Random player for testing purposes (single-player game).
    
    Selects a random valid move at each turn. This is a simple baseline player
    for testing the game logic. For single-player games, it always uses player=1.
    
    Useful for:
    - Basic smoke testing (does the game run without crashing?)
    - Testing that valid moves work correctly
    - Quick sanity checks
    - Baseline comparison for MCTS/AlphaZero performance
    """
    def __init__(self, game):
        """
        Initialize the random player.
        
        Args:
            game: RewritePuzzleGame instance
        """
        self.game = game

    def play(self, board):
        """
        Play a random valid move (single-player game).
        
        This method selects a random action from the set of valid moves.
        For single-player games, player is always 1.
        
        Args:
            board: Current board state (numpy array)
            
        Returns:
            int: Action index to play
        """
        a = np.random.randint(self.game.getActionSize())
        # For single-player games, player is always 1
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a

