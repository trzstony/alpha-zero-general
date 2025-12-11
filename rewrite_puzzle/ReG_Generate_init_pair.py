import random
import sys
import os

# Add the project root to path so we can import rewrite_puzzle modules
# This assumes the script is located in alpha-zero-general/rewrite_puzzle/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from rewrite_puzzle.RewritePuzzleLogic import ExpressionNode, RewritePuzzleBoard
except ImportError as e:
    print(e)
    # Fallback for when running directly from the folder without package structure
    from RewritePuzzleLogic import ExpressionNode, RewritePuzzleBoard

def generate_start_expr(max_depth, is_top_level=True):
    """
    Generates a random ExpressionNode tree with maximum depth max_depth.
    Leaves are integers in {1..9}.
    Internal nodes are '+' or '*'.
    Probabilities: 1/3 for '+', 1/3 for '*', 1/3 for leaf (if not at max depth).
    At top level (is_top_level=True), forces an operator to ensure min depth 2.
    """
    if max_depth == 0:
        return ExpressionNode(value=random.randint(1, 9))
    
    # 0: '+', 1: '*', 2: leaf
    # We use random.choice to ensure uniform probability (1/3 each)
    if is_top_level:
        # Force operator at top level to ensure depth >= 2 (at least one operator)
        choices = ['+', '*']
    else:
        choices = ['+', '*', 'leaf']
        
    choice = random.choice(choices)
    
    if choice == 'leaf':
        return ExpressionNode(value=random.randint(1, 9))
    elif choice == '+':
        left = generate_start_expr(max_depth - 1, is_top_level=False)
        right = generate_start_expr(max_depth - 1, is_top_level=False)
        return ExpressionNode(operator='+', left=left, right=right)
    else: # '*'
        left = generate_start_expr(max_depth - 1, is_top_level=False)
        right = generate_start_expr(max_depth - 1, is_top_level=False)
        return ExpressionNode(operator='*', left=left, right=right)

def generate_goal_expr(start_expr, max_rules):
    """
    Mutates start_expr by applying a sequence of up to max_rules rules.
    
    Args:
        start_expr: The starting ExpressionNode.
        max_rules: Maximum number of rules to apply.
        
    Returns:
        The mutated ExpressionNode (goal_expr).
    """
    # Create a board to manage rules and moves. 
    # Goal expr doesn't matter for generating valid moves, so use start_expr as placeholder.
    # We clone start_expr to avoid modifying the original
    current_expr = start_expr.copy()
    board = RewritePuzzleBoard(current_expr, current_expr)
    
    for _ in range(max_rules):
        valid_actions = board.get_all_valid_actions()
        if not valid_actions:
            break
            
        # Decision: Stop (Do nothing) or Pick a Rule?
        # Requirement: "chosen rule should be draw uniformly (also include the choice of doing nothing, but with less probability)"
        # We assign weight 1.0 to each specific valid action, and a smaller weight to "do nothing".
        
        stop_weight = 0.5  # Less than 1.0, so less probable than any single specific action
        action_weight = 1.0
        total_weight = len(valid_actions) * action_weight + stop_weight
        
        r = random.uniform(0, total_weight)
        
        if r < stop_weight:
            # Do nothing -> Stop mutating
            break
        else:
            # Pick an action uniformly
            # Subtract stop_weight to map r to [0, len(valid_actions) * action_weight)
            action_val = r - stop_weight
            action_idx = int(action_val // action_weight)
            
            # Clamp index just in case of floating point edge cases
            action_idx = min(action_idx, len(valid_actions) - 1)
            
            rule_idx, path = valid_actions[action_idx]
            board.apply_action(rule_idx, path)
            
    return board.current_expr

if __name__ == "__main__":
    # Test the functions
    print("Testing Generation...")
    
    # Generate random start expression
    depth = 3
    start = generate_start_expr(depth)
    print(f"\nGenerated Start Expression (depth={depth}):")
    print(start)
    
    # Generate goal expression
    max_rules = 5
    goal = generate_goal_expr(start, max_rules)
    print(f"\nGenerated Goal Expression (max_rules={max_rules}):")
    print(goal)
    
    # Verify they are different (usually)
    if str(start) != str(goal):
        print("\nSuccess: Goal is different from Start.")
    else:
        print("\nNote: Goal is same as Start (could happen if 'do nothing' chosen or no rules applicable).")
