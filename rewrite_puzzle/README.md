# Rewrite Puzzle Game - AlphaZero Implementation

This is an implementation of a "proof as a game" puzzle where you start with an arithmetic expression and need to rewrite it to match a goal expression using rewrite rules.

## Game Description

**Objective**: Transform a starting arithmetic expression into a goal expression by applying rewrite rules.

**Example**:
- Start: `1 + 2 * 3`
- Goal: `7`
- Rules: 
  - `a + 0 → a`
  - `0 + a → a`
  - `a * 1 → a`
  - `1 * a → a`
  - `a + b → b + a` (commutativity)
  - `a + (b + c) → (a + b) + c` (associativity)
  - `a * (b + c) → a*b + a*c` (distribution)

**State**: Current expression tree + goal expression
**Actions**: Choose a rewrite rule and a location where it applies
**Terminal Conditions**:
- Success: expression equals goal (reward = +1)
- Failure: max steps reached (reward = 0)

## Step-by-Step Implementation Guide

### Step 1: Understand the Structure

The implementation follows the standard AlphaZero-General structure:

```
rewrite_puzzle/
├── __init__.py
├── RewritePuzzleLogic.py      # Core game logic (expression trees, rules)
├── RewritePuzzleGame.py        # Game interface (implements Game.py)
├── RewritePuzzlePlayers.py    # Random and human players
├── pytorch/
│   ├── __init__.py
│   ├── NNet.py                 # Neural network wrapper
│   └── RewritePuzzleNNet.py   # PyTorch neural network architecture
└── main-rewrite-puzzle.py     # Training script
```

### Step 2: Core Components

#### 2.1 Expression Tree Representation (`RewritePuzzleLogic.py`)

- **ExpressionNode**: Represents nodes in the expression tree
  - Leaf nodes: numbers (value)
  - Internal nodes: operators ('+', '*') with left/right children
  - Methods: `evaluate()`, `copy()`, `get_all_subexpressions()`

- **RewriteRule**: Represents a rewrite rule
  - Pattern matching
  - Rule application
  - Rules include: identity, commutativity, associativity, distribution

- **RewritePuzzleBoard**: Game state
  - Current expression
  - Goal expression
  - Steps taken
  - Valid actions enumeration

#### 2.2 Game Interface (`RewritePuzzleGame.py`)

Implements the `Game` base class with:

- **State Encoding**: Expression trees are serialized (pickle + base64) into numpy arrays
- **Action Encoding**: Actions are (rule_idx, position) pairs flattened to a single integer
- **Valid Moves**: Enumerates all valid (rule, location) pairs
- **Terminal Check**: Checks if expression equals goal or max steps reached

#### 2.3 Neural Network (`pytorch/RewritePuzzleNNet.py`)

- Fully connected network (since state is 1D)
- Input: encoded board state (1D array)
- Output: 
  - Policy: probability distribution over actions
  - Value: expected outcome (-1 to +1)

### Step 3: How to Use

#### 3.1 Training

```bash
cd /Volumes/Files/Research\ New\ chapter/alpha-zero-general
python rewrite_puzzle/main-rewrite-puzzle.py
```

The training script will:
1. Initialize the game with default parameters
2. Run self-play episodes
3. Train the neural network
4. Save checkpoints in `./temp/rewrite_puzzle/`

#### 3.2 Customizing the Game

Edit `main-rewrite-puzzle.py` to change:
- Starting expression: `start_expr="1 + 2 * 3"`
- Goal expression: `goal_expr=7`
- Max steps: `max_steps=20`

#### 3.3 Training Parameters

Edit the `args` dictionary in `main-rewrite-puzzle.py`:
- `numIters`: Number of training iterations
- `numEps`: Self-play games per iteration
- `numMCTSSims`: MCTS simulations per move
- `tempThreshold`: Exploration temperature threshold

### Step 4: Key Design Decisions

#### 4.1 Single-Player Adaptation

This is a single-player puzzle, but AlphaZero-General expects two-player games. We adapt by:
- Always using player=1 (the framework still alternates, but it doesn't matter)
- Reward is +1 for success, -1 for failure (max steps)

#### 4.2 State Representation

Challenge: Expression trees are variable-size, but neural networks need fixed-size inputs.

Solution: 
- Serialize expression tree using pickle
- Encode as base64 string
- Store as normalized ASCII values in numpy array
- Fixed size: `max_expr_length + 2` (expression + steps + goal)

#### 4.3 Action Space

Challenge: Actions are (rule, location) pairs, but we need a flat action space.

Solution:
- Flatten: `action = rule_idx * max_positions + position_idx`
- `max_positions` is estimated from expression size
- Valid moves mask ensures only valid actions are considered

### Step 5: Extending the Implementation

#### 5.1 Adding New Rules

Edit `RewritePuzzleBoard._initialize_rules()`:

```python
rules.append(RewriteRule("new_rule_name", "pattern", "replacement"))
```

#### 5.2 Improving Pattern Matching

The current pattern matching is simplified. To improve:
- Implement proper term rewriting with variable matching
- Use a pattern matching library (e.g., `matchpy` for Python)
- Support more complex patterns

#### 5.3 Better State Encoding

Current encoding uses pickle serialization. Alternatives:
- Tree-to-sequence encoding (e.g., preorder traversal)
- Graph neural networks for tree structures
- Attention-based encoders

#### 5.4 Expression Parser

The current parser is basic. Consider:
- Using `ast` module for proper parsing
- Supporting more operators (subtraction, division)
- Handling parentheses correctly

### Step 6: Testing

Test individual components:

```python
from rewrite_puzzle.RewritePuzzleLogic import RewritePuzzleBoard, ExpressionNode

# Create a board
board = RewritePuzzleBoard("1 + 2 * 3", 7, max_steps=20)

# Check valid actions
valid_actions = board.get_all_valid_actions()
print(f"Valid actions: {valid_actions}")

# Apply an action
if valid_actions:
    rule_idx, path = valid_actions[0]
    board.apply_action(rule_idx, path)
    print(f"New expression: {board.current_expr}")
```

### Step 7: Playing Against the Trained Model

Create a play script (similar to other games' `pit.py`):

```python
from rewrite_puzzle.RewritePuzzleGame import RewritePuzzleGame
from rewrite_puzzle.pytorch.NNet import NNetWrapper
from rewrite_puzzle.RewritePuzzlePlayers import HumanRewritePuzzlePlayer
from Arena import Arena

game = RewritePuzzleGame()
nnet = NNetWrapper(game)
nnet.load_checkpoint('./temp/rewrite_puzzle/', 'best.pth.tar')

human = HumanRewritePuzzlePlayer(game)
# ... set up arena and play
```

## Troubleshooting

### Issue: State encoding/decoding fails
- **Solution**: Increase `max_expr_length` or improve serialization

### Issue: Too many invalid actions
- **Solution**: Improve action encoding or use a more sophisticated action space

### Issue: Training doesn't converge
- **Solution**: 
  - Start with simpler expressions
  - Increase `numMCTSSims` for better exploration
  - Adjust learning rate and network architecture

### Issue: Expression parser fails
- **Solution**: Use a proper parser (e.g., `ast.literal_eval` or a custom parser)

## Next Steps

1. **Improve Pattern Matching**: Implement proper term rewriting
2. **Better State Encoding**: Use tree neural networks or better serialization
3. **More Rules**: Add more arithmetic rewrite rules
4. **Visualization**: Create a visual interface for playing
5. **Evaluation**: Test on a suite of puzzles with known solutions

## References

- AlphaZero-General framework: https://github.com/suragnair/alpha-zero-general
- Term Rewriting Systems: Standard topic in formal methods
- Lean Theorem Prover: Inspiration for "proof as a game" concept

