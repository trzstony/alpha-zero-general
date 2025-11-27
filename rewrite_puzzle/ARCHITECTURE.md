# How AlphaZero Works in This Game - Architecture Overview

## The Three Files and Their Roles

### 1. `RewritePuzzleLogic.py` - Game Domain Logic
**Purpose**: Pure game logic, no AlphaZero knowledge

This file defines:
- **ExpressionNode**: How expressions are represented (trees)
- **RewriteRule**: What rewrite rules exist and how they work
- **RewritePuzzleBoard**: Game state management

**Key point**: This file knows NOTHING about AlphaZero, neural networks, or MCTS. It's just the game rules.

### 2. `RewritePuzzleGame.py` - AlphaZero Interface
**Purpose**: Bridge between game logic and AlphaZero framework

This file:
- **Inherits from `Game`** (the base class from the framework)
- **Implements required methods** that AlphaZero needs:
  - `getInitBoard()` - Start state
  - `getNextState()` - Apply an action
  - `getValidMoves()` - What moves are legal
  - `getGameEnded()` - Is the game over?
  - `getCanonicalForm()` - Normalize state
  - `stringRepresentation()` - Hash key for MCTS

**Key point**: This is the ADAPTER that makes your game work with AlphaZero.

### 3. `RewritePuzzlePlayers.py` - Testing Utilities
**Purpose**: Optional testing players (not used in training)

**Key point**: Only for testing, not needed for AlphaZero training.

---

## How AlphaZero Uses These Files

### The Training Flow (in `main-rewrite-puzzle.py`):

```
1. Create Game object (RewritePuzzleGame)
   └─> Uses RewritePuzzleLogic internally
   
2. Create Neural Network (NNetWrapper)
   └─> Needs game.getBoardSize() and game.getActionSize()
   
3. Create Coach (handles training loop)
   └─> Uses Game + Neural Network + MCTS
```

### During Self-Play (Coach.executeEpisode):

```
┌─────────────────────────────────────────────────────────┐
│ 1. Start: game.getInitBoard()                           │
│    └─> RewritePuzzleGame calls RewritePuzzleLogic       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 2. MCTS.search() - For each simulation:                 │
│    a) Check: game.getGameEnded()                        │
│    b) Get: game.getValidMoves()                         │
│    c) Neural Network predicts: (policy, value)          │
│    d) Select action using UCB formula                   │
│    e) Apply: game.getNextState()                        │
│    f) Repeat until terminal state                       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Choose action from MCTS policy distribution          │
│    └─> game.getNextState(board, player, action)         │
│        └─> RewritePuzzleGame calls RewritePuzzleLogic   │
│            to apply rewrite rule                        │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Repeat until game.getGameEnded() != 0                │
│    └─> Collect (state, policy, value) for training      │
└─────────────────────────────────────────────────────────┘
```

### The Neural Network Training:

```
Training Examples: [(board_state, MCTS_policy, game_outcome), ...]
         ↓
Neural Network learns:
  - Policy: Which actions are good in each state
  - Value: How good is this state (closer to solving = better)
         ↓
Next iteration: Use improved network for better MCTS
```

---

## Detailed Connection Points

### MCTS (Monte Carlo Tree Search) uses:

1. **`game.getGameEnded(board, player)`**
   - From: `RewritePuzzleGame.getGameEnded()`
   - Uses: `RewritePuzzleBoard.is_solved()` and `is_terminal()`
   - Returns: 0 (ongoing), 1 (won), -1 (lost)

2. **`game.getValidMoves(board, player)`**
   - From: `RewritePuzzleGame.getValidMoves()`
   - Uses: `RewritePuzzleBoard.get_all_valid_actions()`
   - Returns: Binary vector of valid actions

3. **`game.getNextState(board, player, action)`**
   - From: `RewritePuzzleGame.getNextState()`
   - Uses: `RewritePuzzleBoard.apply_action()`
   - Returns: New board state

4. **`game.stringRepresentation(board)`**
   - From: `RewritePuzzleGame.stringRepresentation()`
   - Used by MCTS for caching/search tree

### Neural Network uses:

1. **`game.getBoardSize()`**
   - From: `RewritePuzzleGame.getBoardSize()`
   - Tells network input dimensions

2. **`game.getActionSize()`**
   - From: `RewritePuzzleGame.getActionSize()`
   - Tells network output dimensions (policy vector size)

3. **`game.getCanonicalForm(board, player)`**
   - From: `RewritePuzzleGame.getCanonicalForm()`
   - Normalizes state for network input

---

## Why This Architecture?

### Separation of Concerns:

```
RewritePuzzleLogic.py
  └─> "What is the game?" (domain knowledge)
      - Expression trees
      - Rewrite rules
      - Game state

RewritePuzzleGame.py  
  └─> "How does AlphaZero interact with the game?" (adapter)
      - State encoding/decoding
      - Action encoding/decoding
      - Framework interface

Framework (Coach, MCTS, NeuralNet)
  └─> "How to learn?" (algorithm)
      - Self-play
      - Tree search
      - Neural network training
```

### Benefits:

1. **Game logic is reusable** - Can use `RewritePuzzleLogic` without AlphaZero
2. **Framework is generic** - Works with any game that implements `Game` interface
3. **Easy to test** - Can test game logic independently
4. **Easy to modify** - Change game rules without touching AlphaZero code

---

## Example: What Happens When AlphaZero Makes a Move

```python
# 1. MCTS searches the game tree
mcts.search(current_board)
  └─> Calls game.getValidMoves()  # What can I do?
  └─> Calls neural_network.predict()  # What does the network think?
  └─> Calls game.getNextState()  # What happens if I do this?
  └─> Repeats many times (numMCTSSims)

# 2. MCTS returns a policy (probability distribution over actions)
policy = mcts.getActionProb(current_board)
  └─> Based on visit counts from search

# 3. Choose an action (with exploration)
action = sample from policy

# 4. Apply the action
new_board, next_player = game.getNextState(current_board, player, action)
  └─> RewritePuzzleGame.getNextState()
      └─> RewritePuzzleBoard.apply_action(rule_idx, path)
          └─> RewriteRule._build_replacement()
              └─> ExpressionNode manipulation

# 5. Check if game ended
result = game.getGameEnded(new_board, next_player)
  └─> RewritePuzzleBoard.is_solved() or steps >= max_steps
```

---

## Summary

- **RewritePuzzleLogic.py**: Your game (domain-specific)
- **RewritePuzzleGame.py**: Adapter to make your game work with AlphaZero (framework-specific)
- **RewritePuzzlePlayers.py**: Testing utilities (optional)

AlphaZero (MCTS + Neural Network) uses the `Game` interface methods to:
1. Understand the game state
2. Explore possible moves
3. Learn which moves lead to winning
4. Improve over time through self-play

The framework doesn't know or care about expression trees or rewrite rules - it just calls the methods you implement in `RewritePuzzleGame.py`, which in turn uses `RewritePuzzleLogic.py` to actually play the game.

