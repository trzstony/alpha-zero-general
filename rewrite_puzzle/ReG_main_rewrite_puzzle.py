"""
Main training script for Rewrite Puzzle game using AlphaZero (single-player version).
This script uses the ReG (single-player) versions of Coach, MCTS, and Arena.
"""

import logging
import sys
import os

# Add parent directory to path to access Game.py and utils.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import using package-style imports to support relative imports in RewritePuzzleGame
from rewrite_puzzle.ReG_Coach import Coach
from rewrite_puzzle.RewritePuzzleGame import RewritePuzzleGame as Game
from rewrite_puzzle.pytorch.NNet import NNetWrapper as nn
from rewrite_puzzle.ReG_Generate_init_pair import generate_start_expr, generate_goal_expr
from utils import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

args = dotdict({
    'numIters': 20,              # Number of training iterations
    'numEps': 10,                  # Number of complete self-play games per iteration
    'tempThreshold': 10,          # Temperature threshold for exploration
    'updateThreshold': 0.55,       # Win rate threshold to accept new model
    'maxlenOfQueue': 20000,        # Maximum number of game examples in queue
    'numMCTSSims': 25,             # Number of MCTS simulations per move
    'arenaCompare': 20,            # Number of games for arena comparison
    'cpuct': 1,                    # Exploration constant for MCTS

    'checkpoint': './temp/rewrite_puzzle/',
    'load_model': True,
    'load_folder_file': ('./temp/rewrite_puzzle/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 10,
})


def randomize_puzzle(game, iter_num):
    """Callback to randomize the puzzle before each iteration."""
    print(f"Randomizing puzzle for iteration {iter_num}...")
    



    # Use specific example for the first iteration and every 10th iteration
    if iter_num == 1 or iter_num % 10 == 0:
        start_expr = "((1 + 2) + 3) + 4"
        goal_expr = "4 + (3 + (2 + 1))"
    else:
        depth = 3
        max_rules = 5
        
        # Generate puzzle until start != goal
        for _ in range(100):  # Safety break after 100 attempts
            start_expr = generate_start_expr(depth)
            goal_expr = generate_goal_expr(start_expr, max_rules)
            if str(start_expr) != str(goal_expr):
                break
        else:
            # Fallback if loop completes without success
            print("Warning: Could not generate distinct start/goal pair after 100 attempts. Using default.")
            start_expr = "((1 + 2) + 3) + 4"
            goal_expr = "4 + (3 + (2 + 1))"
            




    print(f"  Start: {start_expr}")
    print(f"  Goal:  {goal_expr}")
    game.reset_puzzle(start_expr, goal_expr)

def main():
    log.info('Loading %s...', Game.__name__)
    # Initialize game with a placeholder (will be randomized)
    g = Game(start_expr="1 + 2", goal_expr="2 + 1", max_steps=20)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args, on_iteration_start=randomize_puzzle)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
