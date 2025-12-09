
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

from rewrite_puzzle.RewritePuzzleLogic import RewritePuzzleBoard

board = RewritePuzzleBoard("1 + 2", "2 + 1")
print(f"Number of rules: {len(board.rules)}")
for i, rule in enumerate(board.rules):
    print(f"Rule {i}: {rule.name}")

valid_actions = board.get_all_valid_actions()
print(f"\nValid actions ({len(valid_actions)}):")
for rule_idx, path in valid_actions:
    print(f"  Rule idx: {rule_idx}, Path: {path}, len(path): {len(path)}")
    
    # Simulate getValidMoves logic
    max_expr_length = 200
    max_positions = max_expr_length // 2
    position_idx = len(path) % max_positions
    action = rule_idx * max_positions + position_idx
    print(f"  Calculated action: {action} (max_pos={max_positions})")

