# Quick Start Guide - Rewrite Puzzle Game

## Overview

This implementation adapts the AlphaZero-General framework for a single-player rewrite puzzle game where you transform arithmetic expressions using rewrite rules.

## Files Created

1. **RewritePuzzleLogic.py** - Core game logic (expression trees, rules, board state)
2. **RewritePuzzleGame.py** - Game interface implementation
3. **RewritePuzzlePlayers.py** - Random and human players
4. **pytorch/RewritePuzzleNNet.py** - PyTorch neural network architecture
5. **pytorch/NNet.py** - Neural network wrapper
6. **main-rewrite-puzzle.py** - Training script

## Quick Start

### 1. Test the Basic Game Logic

```python
from rewrite_puzzle.RewritePuzzleLogic import RewritePuzzleBoard

# Create a simple puzzle
board = RewritePuzzleBoard("1 + 2 * 3", 7, max_steps=20)
print(f"Start: {board.current_expr}")
print(f"Goal: {board.goal_expr}")

# Get valid actions
valid_actions = board.get_all_valid_actions()
print(f"\nValid actions: {len(valid_actions)}")
for i, (rule_idx, path) in enumerate(valid_actions[:5]):  # Show first 5
    rule = board.rules[rule_idx]
    print(f"  {i}: {rule.name} at {path}")

# Apply an action
if valid_actions:
    rule_idx, path = valid_actions[0]
    board.apply_action(rule_idx, path)
    print(f"\nAfter action: {board.current_expr}")
    print(f"Solved: {board.is_solved()}")
```

### 2. Run Training

```bash
cd /Volumes/Files/Research\ New\ chapter/alpha-zero-general
python rewrite_puzzle/main-rewrite-puzzle.py
```

### 3. Customize the Puzzle

Edit `main-rewrite-puzzle.py`:

```python
g = Game(
    start_expr="2 * 3 + 4",  # Your starting expression
    goal_expr=10,            # Your goal value
    max_steps=15             # Maximum rewrite steps
)
```

## Key Implementation Details

### State Representation
- Expression trees are serialized (pickle + base64) into fixed-size numpy arrays
- Size: `max_expr_length + 2` (expression + steps + goal value)

### Action Space
- Actions are (rule_idx, position) pairs
- Flattened to: `action = rule_idx * max_positions + position_idx`
- Valid moves mask ensures only applicable rules are considered

### Rewrite Rules
Currently implemented:
1. `a + 0 → a` (add_zero_left)
2. `0 + a → a` (add_zero_right)
3. `a * 1 → a` (mult_one_left)
4. `1 * a → a` (mult_one_right)
5. `a + b → b + a` (commutativity)
6. `a + (b + c) → (a + b) + c` (associativity)
7. `a * (b + c) → a*b + a*c` (distribution)

## Example Puzzle

**Start**: `1 + 2 * 3`  
**Goal**: `7`

**Solution path** (one possible):
1. Apply distribution: `1 + 2 * 3` → `1 + (2 * 3)` (if we had that rule)
2. Or evaluate: `2 * 3 = 6`, then `1 + 6 = 7`

Actually, `1 + 2 * 3` evaluates to `7` directly (multiplication first), so the goal is already met if we consider evaluation order. But with rewrite rules, we might need to transform it.

## Next Steps

1. **Test the implementation**: Run the training script and see if it works
2. **Adjust parameters**: Modify training hyperparameters in `main-rewrite-puzzle.py`
3. **Add more rules**: Extend `_initialize_rules()` in `RewritePuzzleLogic.py`
4. **Improve parser**: The current parser is basic - consider using `ast` module
5. **Better state encoding**: Consider tree neural networks for better representation

## Troubleshooting

- **Import errors**: Make sure you're running from the repo root directory
- **State encoding fails**: Increase `max_expr_length` in Game initialization
- **No valid actions**: Check that your expression can actually be transformed
- **Training issues**: Start with simpler expressions and fewer rules

## See Also

- `README.md` for detailed documentation
- Other game implementations (e.g., `tictactoe/`, `othello/`) for reference

