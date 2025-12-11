"""
Game class implementation for the Rewrite Puzzle game.
This is a single-player puzzle game where you rewrite expressions to match a goal.
"""

from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .RewritePuzzleLogic import RewritePuzzleBoard, ExpressionNode
import numpy as np


class RewritePuzzleGame(Game):
    def __init__(self, start_expr="1 + 2 * 3", goal_expr=7, max_steps=20, max_expr_length=200):
        """
        Initialize the rewrite puzzle game.
        
        Args:
            start_expr: Starting expression (string or ExpressionNode)
            goal_expr: Goal expression (string, ExpressionNode, or number)
            max_steps: Maximum number of rewrite steps allowed
            max_expr_length: Maximum length for expression encoding
        """
        self.start_expr = start_expr
        self.goal_expr = goal_expr
        self.max_steps = max_steps
        self.max_expr_length = max_expr_length
        self.num_rules = 5  # Number of rewrite rules (updated to match actual rule count)
        
        # Initialize the board
        self.board = RewritePuzzleBoard(start_expr, goal_expr, max_steps)
        
    def reset_puzzle(self, start_expr, goal_expr):
        """Reset the game with a new puzzle."""
        self.start_expr = start_expr
        self.goal_expr = goal_expr
        self.board = RewritePuzzleBoard(start_expr, goal_expr, self.max_steps)

    def getInitBoard(self):
        """Returns initial board state as a numpy array."""
        return self._encode_state(self.board)
    
    def getBoardSize(self):
        """Returns board dimensions."""
        # We encode: serialized board state (max_expr_length bytes as floats)
        # Plus steps taken (1), plus goal value (1)
        return (self.max_expr_length + 2,)
    
    def getActionSize(self):
        """Returns number of all possible actions."""
        # Action = (rule_idx, position_idx)
        # We'll use a flattened representation: rule_idx * max_positions + position_idx
        # For simplicity, we'll use a fixed max_positions based on max_expr_length
        max_positions = self.max_expr_length // 2  # Rough estimate
        return self.num_rules * max_positions
    
    def getNextState(self, board, player, action):
        """
        Returns next state after applying action.
        Note: This is a single-player game, so player alternation is not meaningful.
        """
        # Decode board state
        board_obj = self._decode_state(board)
        
        # Decode action
        max_positions = self.max_expr_length // 2
        rule_idx = action // max_positions
        position_idx = action % max_positions
        
        # Get valid actions and apply if valid
        valid_actions = board_obj.get_all_valid_actions()
        
        # Find matching action by both rule_idx AND position_idx
        # The action encoding uses: action = rule_idx * max_positions + position_idx
        # where position_idx = len(path) % max_positions
        applied = False
        
        # First, try to find exact match by rule_idx and position_idx
        matching_actions = []
        for valid_rule_idx, valid_path in valid_actions:
            if valid_rule_idx == rule_idx:
                valid_position_idx = len(valid_path) % max_positions
                if valid_position_idx == position_idx:
                    matching_actions.append((valid_rule_idx, valid_path))
        
        # Try to apply matching actions (in order)
        for valid_rule_idx, valid_path in matching_actions:
            if board_obj.apply_action(valid_rule_idx, valid_path):
                applied = True
                break
        
        
        # If no exact match, try any action with matching rule_idx (fallback)
        if not applied:
            for valid_rule_idx, valid_path in valid_actions:
                if valid_rule_idx == rule_idx:
                    if board_obj.apply_action(valid_rule_idx, valid_path):
                        applied = True
                        break
        
        # If action wasn't valid, just increment steps (penalty)
        if not applied:
            board_obj.steps_taken += 1
        
        # Encode new state
        new_board = self._encode_state(board_obj)
        # In single-player games, we don't really alternate players
        # But the framework expects it, so we'll return -player
        return (new_board, -player)
    
    def getValidMoves(self, board, player):
        """Returns binary vector of valid moves."""
        board_obj = self._decode_state(board)
        valid_actions = board_obj.get_all_valid_actions()
        
        max_positions = self.max_expr_length // 2
        valids = np.zeros(self.getActionSize(), dtype=np.int32)
        
        for rule_idx, path in valid_actions:
            # Encode position as a simple index (simplified)
            position_idx = len(path) % max_positions
            action = rule_idx * max_positions + position_idx
            if action < len(valids):
                valids[action] = 1
        
        # If no valid moves, allow pass (last action)
        if np.sum(valids) == 0:
            valids[-1] = 1
        
        return valids
    
    def getGameEnded(self, board, player):
        """
        Returns:
            0 if game not ended
            1 if player won (solved the puzzle)
            -1 if player lost (max steps reached without solving)
        """
        board_obj = self._decode_state(board)
        
        if board_obj.is_solved():
            return 1
        elif board_obj.is_terminal() and not board_obj.is_solved():
            return -1
        return 0
    
    def getCanonicalForm(self, board, player):
        """Returns canonical form of board (same for single-player game)."""
        return board
    
    def getSymmetries(self, board, pi):
        """Returns symmetrical forms (limited for expression trees)."""
        # For expressions, symmetries are limited
        # We could consider commutativity symmetries, but for simplicity return as-is
        return [(board, pi)]
    
    def stringRepresentation(self, board):
        """Returns string representation for hashing."""
        board_obj = self._decode_state(board)
        return str(board_obj.current_expr) + "|" + str(board_obj.goal_expr) + "|" + str(board_obj.steps_taken)
    
    def _encode_state(self, board_obj):
        """Encode board state as numpy array using string representation.
        
        Format:
        - Slots 0 to (max_expr_length-3): Current expression string (ASCII codes / 128.0)
        - Slot max_expr_length-2: Steps taken (normalized to [0, 1])
        - Slot max_expr_length-1: Goal expression string length (for parsing)
        - Remaining slots: Goal expression string (ASCII codes / 128.0)
        """
        state = np.zeros(self.getBoardSize()[0], dtype=np.float32)
        
        # Convert expressions to strings
        current_expr_str = str(board_obj.current_expr)
        
        # Convert goal to string
        if isinstance(board_obj.goal_expr, ExpressionNode):
            goal_expr_str = str(board_obj.goal_expr)
        elif isinstance(board_obj.goal_expr, (int, float)):
            goal_expr_str = str(int(board_obj.goal_expr))
        else:
            goal_expr_str = str(board_obj.goal_expr)
        
        # Calculate available space
        # Reserve: 2 slots for steps and goal_length, rest for expressions
        reserved_slots = 2
        expr_slots = self.max_expr_length - reserved_slots
        
        # Split space between current_expr and goal_expr (70% for current, 30% for goal)
        current_expr_max_len = int(expr_slots * 0.7)
        goal_expr_max_len = expr_slots - current_expr_max_len
        
        # Encode current expression
        current_expr_encoded = current_expr_str[:current_expr_max_len]
        for i, char in enumerate(current_expr_encoded):
            if i < current_expr_max_len:
                state[i] = ord(char) / 128.0
        
        # Encode goal expression
        goal_expr_encoded = goal_expr_str[:goal_expr_max_len]
        goal_start_idx = current_expr_max_len
        for i, char in enumerate(goal_expr_encoded):
            if i < goal_expr_max_len:
                state[goal_start_idx + i] = ord(char) / 128.0
        
        # Store metadata at the end
        # Slot -2: Steps taken (normalized)
        state[-2] = board_obj.steps_taken / float(self.max_steps)
        
        # Slot -1: Goal expression length (for decoding)
        state[-1] = len(goal_expr_encoded) / 1000.0  # Normalize (assuming max length < 1000)
        
        return state
    
    def _decode_state(self, board_array):
        """Decode numpy array back to board object.
        
        Format matches _encode_state:
        - Slots 0 to (max_expr_length-3): Current expression string
        - Slot max_expr_length-2: Steps taken
        - Slot max_expr_length-1: Goal expression length
        - Remaining slots: Goal expression string
        """
        reserved_slots = 2
        expr_slots = self.max_expr_length - reserved_slots
        current_expr_max_len = int(expr_slots * 0.7)
        goal_expr_max_len = expr_slots - current_expr_max_len
        
        # Decode current expression string
        current_expr_chars = []
        for i in range(current_expr_max_len):
            val = board_array[i]
            if val > 0:
                char_code = int(val * 128.0)
                if 32 <= char_code < 128:  # Valid ASCII
                    current_expr_chars.append(chr(char_code))
                elif char_code == 0:
                    break  # End of string
            else:
                break  # End of string
        
        # Decode goal expression string
        goal_expr_chars = []
        goal_start_idx = current_expr_max_len
        goal_length = int(board_array[-1] * 1000.0)  # Denormalize
        goal_length = min(goal_length, goal_expr_max_len)  # Safety check
        
        for i in range(goal_length):
            val = board_array[goal_start_idx + i]
            if val > 0:
                char_code = int(val * 128.0)
                if 32 <= char_code < 128:  # Valid ASCII
                    goal_expr_chars.append(chr(char_code))
                elif char_code == 0:
                    break
            else:
                break
        
        # Reconstruct expressions
        current_expr_str = ''.join(current_expr_chars)
        goal_expr_str = ''.join(goal_expr_chars)
        
        # Decode steps
        steps = int(board_array[-2] * self.max_steps)
        
        # Create board and parse expressions
        board = RewritePuzzleBoard(self.start_expr, self.goal_expr, self.max_steps)
        
        # Parse current expression
        if current_expr_str:
            try:
                board.current_expr = board._parse_expr(current_expr_str)
            except Exception as e:
                # If parsing fails, use original start_expr
                import logging
                logging.warning(f"Failed to parse current_expr '{current_expr_str}': {e}")
                board.current_expr = board._parse_expr(str(self.start_expr))
        else:
            board.current_expr = board._parse_expr(str(self.start_expr))
        
        # Parse goal expression
        if goal_expr_str:
            try:
                # Try parsing as expression first
                parsed_goal = board._parse_expr(goal_expr_str)
                # Check if it's just a number
                if parsed_goal.is_leaf():
                    board.goal_expr = int(parsed_goal.value)
                else:
                    board.goal_expr = parsed_goal
            except Exception as e:
                # If parsing fails, try as integer
                try:
                    board.goal_expr = int(goal_expr_str)
                except ValueError as e2:
                    print(e2)
                    board.goal_expr = self.goal_expr
        else:
            board.goal_expr = self.goal_expr
        
        board.steps_taken = steps
        
        return board
    
    @staticmethod
    def display(board):
        """Display the board state."""
        # This would need the game instance to decode properly
        print("Board state:", board)
