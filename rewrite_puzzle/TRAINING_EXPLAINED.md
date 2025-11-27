# AlphaZero Training Process Explained with Simple Example

## Setup

**Game**: Start with `1 + 2 * 3`, goal is `7`, max 20 steps

**Training Parameters**:
- `numIters = 100`: Run 100 training iterations
- `numEps = 50`: Play 50 self-play games per iteration
- `numMCTSSims = 25`: Do 25 MCTS simulations per move

---

## What Happens When You Run `main-rewrite-puzzle.py`

### Step 1: Initialization

```python
g = Game(start_expr="1 + 2 * 3", goal_expr=7, max_steps=20)
nnet = nn(g)  # Create neural network (randomly initialized)
c = Coach(g, nnet, args)  # Create coach
```

**What this does:**
- Creates the game with starting expression `1 + 2 * 3`
- Creates a neural network (starts with random weights - knows nothing!)
- Creates a Coach that will manage training

---

## Training Loop: One Iteration in Detail

Let's trace through **Iteration #1**:

### Phase 1: Self-Play (Collect Training Data)

The Coach plays **50 games** against itself:

#### Example: One Self-Play Game

**Initial State**: `1 + 2 * 3` (goal: 7)

**Move 1:**
1. **MCTS explores** (25 simulations):
   - Neural network predicts: "Maybe try rule X at position Y"
   - MCTS simulates many possible futures
   - MCTS returns a **policy** (probability distribution over actions)
     - Example: `[0.1, 0.05, 0.3, 0.2, 0.15, 0.1, 0.1]` (7 possible actions)
     - Action 2 has 30% probability (maybe "commute addition" rule)

2. **Choose action** from policy (with exploration):
   - Randomly sample: Action 2 (30% chance)
   - Apply: `1 + 2 * 3` → `2 * 3 + 1` (commutativity)

3. **Save training example**:
   ```python
   (state="1 + 2 * 3", policy=[0.1, 0.05, 0.3, ...], value=None)
   ```
   (Value will be filled in at the end)

**Move 2:**
- State: `2 * 3 + 1`
- MCTS explores again
- Choose action
- Apply rule
- Save example

**... continues until game ends ...**

**Game Ends** (either solved or max steps):
- If solved: `value = +1` (won!)
- If failed: `value = -1` (lost)

**Backpropagate values**:
- All saved examples get their values:
  - Move 1: `value = +1` (if game was won)
  - Move 2: `value = +1`
  - Move 3: `value = +1`
  - etc.

**Result**: One game produces ~5-10 training examples

**Repeat 50 times**: Now we have ~250-500 training examples from this iteration

---

### Phase 2: Train Neural Network

```python
trainExamples = [all examples from current + previous iterations]
shuffle(trainExamples)  # Randomize order
nnet.train(trainExamples)  # Train the network
```

**What the network learns:**
- **Policy head**: "In state X, action Y is good" (from MCTS policies)
- **Value head**: "State X is good/bad" (from game outcomes)

**Example training step:**
- Input: State `"1 + 2 * 3"`
- Target policy: `[0.1, 0.05, 0.3, ...]` (from MCTS)
- Target value: `+1` (game was won)
- Network adjusts weights to match these targets

---

### Phase 3: Arena Comparison (Is New Model Better?)

```python
# Old model (previous iteration)
pnet = previous_model
pmcts = MCTS(game, pnet, args)

# New model (just trained)
nnet = current_model  
nmcts = MCTS(game, nnet, args)

# Play 20 games: new vs old
arena.playGames(20)
```

**What happens:**
- New model plays 20 games against old model
- Count wins: `nwins` (new) vs `pwins` (previous)

**Decision:**
- If `nwins / (nwins + pwins) >= 0.55` (55% win rate):
  - ✅ **ACCEPT** new model (save as `best.pth.tar`)
- Else:
  - ❌ **REJECT** new model (revert to old one)

**Why?** Prevents the model from getting worse over time.

---

## Complete Training Flow (100 Iterations)

```
Iteration 1:
  ├─ Play 50 self-play games → collect ~500 examples
  ├─ Train network on examples
  ├─ Arena: New vs Old (new is random, so might lose)
  └─ Result: Maybe accept, maybe reject

Iteration 2:
  ├─ Play 50 games (network is slightly better now)
  ├─ Collect examples
  ├─ Train network
  ├─ Arena: New vs Old
  └─ Result: Hopefully new wins more!

Iteration 3:
  ├─ Network is getting smarter...
  └─ ...

... (continues for 100 iterations) ...

Iteration 100:
  ├─ Network is now quite good!
  ├─ Can solve the puzzle reliably
  └─ Final model saved as best.pth.tar
```

---

## Concrete Example: One Complete Game

Let's trace one self-play game in detail:

### Game State Progression

**Start**: `1 + 2 * 3` (value = 7, goal = 7) ✅ Already solved!

Wait, but the network doesn't know this yet. Let's say it's a harder puzzle:

**Start**: `2 * 3 + 1` (value = 7, goal = 7)

**Move 1:**
- State: `2 * 3 + 1`
- MCTS runs 25 simulations:
  - Tries: commute → `3 + 1 * 2` (value = 5, worse!)
  - Tries: commute → `1 + 2 * 3` (value = 7, goal!)
  - Tries: other rules...
  - MCTS learns: "commute to get `1 + 2 * 3` is good"
- Policy: `[0.1, 0.05, 0.7, 0.1, 0.05]` (action 2 = 70%)
- Sample action: 2
- Apply: `2 * 3 + 1` → `1 + 2 * 3`
- Save: `(state="2*3+1", policy=[0.1,0.05,0.7,...], value=None)`

**Move 2:**
- State: `1 + 2 * 3` (value = 7)
- Check: `game.getGameEnded()` → returns `1` (solved!)
- Game ends!

**Assign values:**
- Move 1: `value = +1` (led to win)
- Final example: `(state="2*3+1", policy=[0.1,0.05,0.7,...], value=+1)`

---

## What the Network Learns Over Time

### Iteration 1 (Random Network):
- Makes random moves
- Occasionally solves by luck
- Learns: "Some states are good, some are bad"

### Iteration 10:
- Starts recognizing patterns
- "Oh, when I see `X + 0`, I should simplify to `X`"
- Still makes mistakes

### Iteration 50:
- Much better at choosing good moves
- Recognizes many useful patterns
- Can solve easier puzzles reliably

### Iteration 100:
- Expert level
- Knows which rules to apply when
- Can solve the puzzle efficiently

---

## Key Concepts

### 1. **Self-Play**
- Network plays against itself
- No human examples needed
- Learns from its own experience

### 2. **MCTS (Monte Carlo Tree Search)**
- Explores possible futures
- Balances exploration vs exploitation
- Provides better policies than raw network

### 3. **Policy + Value Learning**
- **Policy**: Which action to take
- **Value**: How good is this state
- Both learned simultaneously

### 4. **Arena Comparison**
- Prevents model degradation
- Only accepts improvements
- Ensures steady progress

---

## Summary

**Training = Play → Learn → Test → Repeat**

1. **Play**: Network plays many games against itself
2. **Learn**: Network trains on game outcomes
3. **Test**: New model competes against old model
4. **Repeat**: Do this 100 times

Each iteration makes the network slightly better, until it becomes an expert at solving the rewrite puzzle!

