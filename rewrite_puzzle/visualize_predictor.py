"""
Visualization script for the Rewrite Puzzle neural network predictor.

This script answers:
1. Where is the predictor (model) stored
2. How does the predictor predict the outcome
3. What is the input of the predictor, what is the output
4. How to visualize the predictor's functionality with real outputs in vector form
"""

import numpy as np
import sys
import os
import torch

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from rewrite_puzzle.RewritePuzzleGame import RewritePuzzleGame
from rewrite_puzzle.pytorch.NNet import NNetWrapper
from rewrite_puzzle.ReG_MCTS import MCTS
from utils import dotdict


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def visualize_model_storage():
    """Answer: Where is the predictor (model) stored?"""
    print_section("1. WHERE IS THE PREDICTOR (MODEL) STORED?")
    
    print("The model is stored as PyTorch checkpoint files (.pth.tar format).")
    print("\nStorage locations:")
    print("  - Default checkpoint folder: './temp/rewrite_puzzle/'")
    print("  - Best model: './temp/rewrite_puzzle/best.pth.tar'")
    print("  - Iteration checkpoints: './temp/rewrite_puzzle/checkpoint_N.pth.tar'")
    print("  - Temporary model: './temp/rewrite_puzzle/temp.pth.tar'")
    
    print("\nModel structure:")
    print("  - The model contains the neural network's state_dict (weights and biases)")
    print("  - Saved using: torch.save({'state_dict': self.nnet.state_dict()}, filepath)")
    print("  - Loaded using: checkpoint = torch.load(filepath); self.nnet.load_state_dict(checkpoint['state_dict'])")
    
    # Check if model exists
    checkpoint_path = './temp/rewrite_puzzle/best.pth.tar'
    if os.path.exists(checkpoint_path):
        print(f"\n✓ Found model at: {checkpoint_path}")
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        print(f"  File size: {file_size:.2f} MB")
    else:
        print(f"\n✗ No model found at: {checkpoint_path}")
        print("  (This is normal if training hasn't been run yet)")


def visualize_prediction_process(game, nnet):
    """Answer: How does the predictor predict the outcome?"""
    print_section("2. HOW DOES THE PREDICTOR PREDICT THE OUTCOME?")
    
    print("The prediction process has two main components:")
    print("\nA. Neural Network Prediction:")
    print("  1. Input: Board state (1D numpy array)")
    print("  2. Forward pass through fully connected layers:")
    print("     - Input → FC1 (256 channels) → BatchNorm → ReLU → Dropout")
    print("     - → FC2 (256) → BatchNorm → ReLU → Dropout")
    print("     - → FC3 (256) → BatchNorm → ReLU → Dropout")
    print("     - → FC4 (256) → BatchNorm → ReLU → Dropout")
    print("     - → FC5 (512) → BatchNorm → ReLU → Dropout")
    print("     - → FC6 (256) → BatchNorm → ReLU → Dropout")
    print("  3. Two output heads:")
    print("     - Policy head (FC_pi): action_size logits → log_softmax → exp")
    print("     - Value head (FC_v): 1 value → tanh")
    
    print("\nB. MCTS Enhancement:")
    print("  1. Neural network provides initial policy (pi) and value (v)")
    print("  2. MCTS performs numMCTSSims simulations (default: 25)")
    print("  3. Each simulation:")
    print("     - Selects action with highest UCB (Upper Confidence Bound)")
    print("     - Expands leaf nodes using neural network predictions")
    print("     - Backpropagates values up the tree")
    print("  4. Final policy: normalized visit counts from MCTS tree")
    
    print("\nCode flow:")
    print("  nnet.predict(board) → (pi_raw, v_raw)")
    print("  mcts.getActionProb(board) → uses nnet.predict internally → returns improved policy")


def visualize_input_output(game, nnet):
    """Answer: What is the input and output of the predictor?"""
    print_section("3. WHAT IS THE INPUT AND OUTPUT OF THE PREDICTOR?")
    
    # Create a sample game state
    game_instance = game(start_expr="1 + (2 * 3)", goal_expr="(3 * 2) + 1", max_steps=20)
    board = game_instance.getInitBoard()
    canonical_board = game_instance.getCanonicalForm(board, 1)
    
    print("INPUT:")
    print(f"  Type: numpy.ndarray")
    print(f"  Shape: {canonical_board.shape}")
    print(f"  Dtype: {canonical_board.dtype}")
    print(f"  Size: {canonical_board.size} elements")
    
    print("\n  Input encoding format:")
    print("    - Slots 0 to ~70% of max_expr_length: Current expression (ASCII codes / 128.0)")
    print("    - Slots ~70% to ~100%: Goal expression (ASCII codes / 128.0)")
    print("    - Slot -2: Steps taken (normalized to [0, 1])")
    print("    - Slot -1: Goal expression length (normalized)")
    
    print(f"\n  Sample input (first 20 values):")
    print(f"    {canonical_board[:20]}")
    print(f"  Sample input (last 5 values):")
    print(f"    {canonical_board[-5:]}")
    
    # Get prediction
    pi_raw, v_raw = nnet.predict(canonical_board)
    
    print("\nOUTPUT:")
    print("  The predictor returns TWO outputs:")
    print("\n  A. Policy Vector (pi):")
    print(f"    Type: numpy.ndarray")
    print(f"    Shape: {pi_raw.shape}")
    print(f"    Dtype: {pi_raw.dtype}")
    print(f"    Size: {pi_raw.size} elements (one per possible action)")
    print(f"    Range: [0, 1] (probabilities, sum to ~1)")
    print(f"    Meaning: Probability distribution over all possible actions")
    print(f"    Processing: log_softmax → exp (ensures probabilities sum to 1)")
    
    print(f"\n    Sample policy (top 10 actions):")
    top_indices = np.argsort(pi_raw)[::-1][:10]
    for i, idx in enumerate(top_indices):
        print(f"      Action {idx:4d}: probability = {pi_raw[idx]:.6f}")
    
    print("\n  B. Value Scalar (v):")
    v_scalar = float(np.array(v_raw).item() if isinstance(v_raw, np.ndarray) else v_raw)
    print(f"    Type: numpy.ndarray (scalar)")
    print(f"    Shape: {np.array(v_raw).shape}")
    print(f"    Value: {v_scalar:.6f}")
    print(f"    Range: [-1, 1] (tanh activation)")
    print(f"    Meaning: Expected outcome from current state")
    print(f"             - For single-player: probability of solving the puzzle")
    print(f"             - +1 = likely to solve, -1 = likely to fail")
    print(f"    Processing: Linear layer → tanh (bounded to [-1, 1])")
    
    return canonical_board, pi_raw, v_raw


def visualize_full_prediction(game, nnet, board_state):
    """Answer: How to visualize the predictor's functionality with real outputs?"""
    print_section("4. VISUALIZING PREDICTOR FUNCTIONALITY - REAL OUTPUTS")
    
    # Decode the board to show what it represents
    board_obj = game._decode_state(board_state)
    print("CURRENT GAME STATE:")
    print(f"  Current expression: {board_obj.current_expr}")
    print(f"  Goal expression: {board_obj.goal_expr}")
    print(f"  Steps taken: {board_obj.steps_taken}/{board_obj.max_steps}")
    print(f"  Is solved: {board_obj.is_solved()}")
    
    # Get raw neural network prediction
    print("\n" + "-"*80)
    print("STEP 1: RAW NEURAL NETWORK PREDICTION")
    print("-"*80)
    pi_raw, v_raw = nnet.predict(board_state)
    v_scalar = float(np.array(v_raw).item() if isinstance(v_raw, np.ndarray) else v_raw)
    
    print(f"\nRaw Policy Vector (pi):")
    print(f"  Shape: {pi_raw.shape}")
    print(f"  Sum: {np.sum(pi_raw):.6f} (should be ~1.0)")
    print(f"  Min: {np.min(pi_raw):.6f}, Max: {np.max(pi_raw):.6f}")
    print(f"  Non-zero entries: {np.count_nonzero(pi_raw)}")
    
    print(f"\nRaw Value (v):")
    print(f"  Value: {v_scalar:.6f}")
    print(f"  Interpretation: {'Likely to solve' if v_scalar > 0 else 'Likely to fail'}")
    
    # Show valid moves
    valids = game.getValidMoves(board_state, 1)
    valid_actions = board_obj.get_all_valid_actions()
    print(f"\nValid Actions: {len(valid_actions)} out of {game.getActionSize()} total actions")
    
    # Mask policy with valid moves
    pi_masked = pi_raw * valids
    if np.sum(pi_masked) > 0:
        pi_masked = pi_masked / np.sum(pi_masked)
    
    print(f"\nMasked Policy (only valid moves):")
    print(f"  Sum: {np.sum(pi_masked):.6f}")
    print(f"  Top 5 valid actions:")
    max_positions = game.max_expr_length // 2
    action_scores = []
    for rule_idx, path in valid_actions:
        position_idx = len(path) % max_positions
        action = rule_idx * max_positions + position_idx
        if action < len(pi_masked):
            prob = pi_masked[action]
            try:
                if hasattr(board_obj, 'rules') and rule_idx < len(board_obj.rules):
                    rule_name = board_obj.rules[rule_idx].name
                else:
                    rule_name = f"Rule{rule_idx}"
            except (AttributeError, IndexError):
                rule_name = f"Rule{rule_idx}"
            action_scores.append((prob, rule_idx, path, rule_name, action))
    
    action_scores.sort(reverse=True, key=lambda x: x[0])
    for i, (prob, rule_idx, path, rule_name, action) in enumerate(action_scores[:5]):
        print(f"    {i+1}. Action {action:4d} ({rule_name:20s}): {prob:.6f}")
    
    # MCTS enhanced prediction
    print("\n" + "-"*80)
    print("STEP 2: MCTS-ENHANCED PREDICTION")
    print("-"*80)
    args = dotdict({
        'numMCTSSims': 25,
        'cpuct': 1,
    })
    mcts = MCTS(game, nnet, args)
    pi_mcts = mcts.getActionProb(board_state, temp=0)
    
    print(f"\nMCTS Policy Vector (pi_mcts):")
    print(f"  Shape: {pi_mcts.shape}")
    print(f"  Sum: {np.sum(pi_mcts):.6f} (should be 1.0)")
    print(f"  Min: {np.min(pi_mcts):.6f}, Max: {np.max(pi_mcts):.6f}")
    print(f"  Non-zero entries: {np.count_nonzero(pi_mcts)}")
    
    print(f"\nTop 5 actions after MCTS:")
    mcts_action_scores = []
    for rule_idx, path in valid_actions:
        position_idx = len(path) % max_positions
        action = rule_idx * max_positions + position_idx
        if action < len(pi_mcts):
            prob = pi_mcts[action]
            try:
                if hasattr(board_obj, 'rules') and rule_idx < len(board_obj.rules):
                    rule_name = board_obj.rules[rule_idx].name
                else:
                    rule_name = f"Rule{rule_idx}"
            except (AttributeError, IndexError):
                rule_name = f"Rule{rule_idx}"
            mcts_action_scores.append((prob, rule_idx, path, rule_name, action))
    
    mcts_action_scores.sort(reverse=True, key=lambda x: x[0])
    for i, (prob, rule_idx, path, rule_name, action) in enumerate(mcts_action_scores[:5]):
        print(f"    {i+1}. Action {action:4d} ({rule_name:20s}): {prob:.6f}")
    
    # Comparison
    print("\n" + "-"*80)
    print("COMPARISON: Raw NN vs MCTS-Enhanced")
    print("-"*80)
    best_raw = np.argmax(pi_masked)
    best_mcts = np.argmax(pi_mcts)
    print(f"\nBest action (Raw NN):    {best_raw} (prob: {pi_masked[best_raw]:.6f})")
    print(f"Best action (MCTS):       {best_mcts} (prob: {pi_mcts[best_mcts]:.6f})")
    print(f"Same action?             {'Yes' if best_raw == best_mcts else 'No'}")
    
    # Full vector output
    print("\n" + "-"*80)
    print("FULL VECTOR OUTPUTS")
    print("-"*80)
    print("\nRaw Policy Vector (first 50 values):")
    print(pi_raw[:50])
    print("\nRaw Policy Vector (last 50 values):")
    print(pi_raw[-50:])
    
    print("\nMCTS Policy Vector (first 50 values):")
    print(pi_mcts[:50])
    print("\nMCTS Policy Vector (last 50 values):")
    print(pi_mcts[-50:])
    
    return pi_raw, v_raw, pi_mcts


def main():
    """Main function to run all visualizations."""
    print("\n" + "="*80)
    print("  REWRITE PUZZLE NEURAL NETWORK PREDICTOR VISUALIZATION")
    print("="*80)
    
    # Initialize game and network
    print("\nInitializing game and neural network...")
    game = RewritePuzzleGame
    game_instance = game(start_expr="1 + (2 * 3)", goal_expr="(3 * 2) + 1", max_steps=20)
    nnet = NNetWrapper(game_instance)
    
    # Check if model exists and try to load it
    checkpoint_path = './temp/rewrite_puzzle/best.pth.tar'
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}...")
        try:
            nnet.load_checkpoint('./temp/rewrite_puzzle/', 'best.pth.tar')
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("  Using untrained model (random weights)")
    else:
        print("No trained model found. Using untrained model (random weights).")
        print("  (To train a model, run ReG_main_rewrite_puzzle.py)")
    
    # Run visualizations
    visualize_model_storage()
    visualize_prediction_process(game, nnet)
    board_state, pi_raw, v_raw = visualize_input_output(game_instance, nnet)
    visualize_full_prediction(game_instance, nnet, board_state)
    
    print("\n" + "="*80)
    print("  VISUALIZATION COMPLETE")
    print("="*80)
    print("\nSummary:")
    print("  1. Models are stored in './temp/rewrite_puzzle/' as .pth.tar files")
    print("  2. Prediction uses a fully connected neural network + MCTS")
    print("  3. Input: 1D array encoding game state, Output: policy vector + value scalar")
    print("  4. Use nnet.predict(board) to get raw predictions, mcts.getActionProb() for MCTS-enhanced")


if __name__ == "__main__":
    main()
