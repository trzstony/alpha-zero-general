"""
Logic for the Rewrite Puzzle Game.
Handles expression trees and rewrite rule application.
"""

import copy
from typing import List, Tuple, Optional, Dict, Any


class ExpressionNode:
    """Represents a node in an expression tree."""
    def __init__(self, value=None, operator=None, left=None, right=None):
        # Convert value to int if provided (ensure only integers)
        if value is not None:
            self.value = int(value)  # Force integer conversion
        else:
            self.value = value
        self.operator = operator  # For internal nodes ('+', '*')
        self.left = left
        self.right = right
    
    def is_leaf(self):
        return self.value is not None
    
    def evaluate(self):
        """Evaluate the expression tree. Returns integer."""
        if self.is_leaf():
            return int(self.value)
        left_val = self.left.evaluate()
        right_val = self.right.evaluate()
        if self.operator == '+':
            return int(left_val + right_val)
        elif self.operator == '*':
            return int(left_val * right_val)
        else:
            raise ValueError(f"Unknown operator: {self.operator}")
    
    def __eq__(self, other):
        """Check if two expressions are structurally equal."""
        if isinstance(other, int):
            return self.is_leaf() and int(self.value) == other
        if isinstance(other, float):
            # Allow float comparison but convert to int for consistency
            return self.is_leaf() and int(self.value) == int(other)
        if not isinstance(other, ExpressionNode):
            return False
        if self.is_leaf() and other.is_leaf():
            return int(self.value) == int(other.value)
        if not self.is_leaf() and not other.is_leaf():
            return (self.operator == other.operator and 
                   self.left == other.left and 
                   self.right == other.right)
        return False
    
    def __str__(self):
        if self.is_leaf():
            return str(self.value)
        return f"({self.left} {self.operator} {self.right})"
    
    def copy(self):
        """Deep copy of the expression tree."""
        if self.is_leaf():
            return ExpressionNode(value=self.value)
        return ExpressionNode(
            operator=self.operator,
            left=self.left.copy(),
            right=self.right.copy()
        )
    
    def get_all_subexpressions(self):
        """Get all subexpressions (nodes) in the tree."""
        result = [self]
        if not self.is_leaf():
            result.extend(self.left.get_all_subexpressions())
            result.extend(self.right.get_all_subexpressions())
        return result
    
    def find_matches(self, pattern, parent=None, path=[]):
        """Find all locations where a pattern matches."""
        matches = []
        if self.matches_pattern(pattern):
            matches.append((self, parent, path))
        if not self.is_leaf():
            matches.extend(self.left.find_matches(pattern, self, path + ['left']))
            matches.extend(self.right.find_matches(pattern, self, path + ['right']))
        return matches
    
    def matches_pattern(self, pattern):
        """Check if this node matches a pattern (simplified pattern matching)."""
        # For now, we'll do simple structural matching
        # In a full implementation, you'd want more sophisticated pattern matching
        if pattern == 'any':
            return True
        if isinstance(pattern, ExpressionNode):
            return self._structural_match(pattern)
        return False

    def find_matches_strict(self, pattern, parent=None, path=[]):
        """Find all locations where a pattern matches with strict value matching."""
        matches = []
        if self.matches_pattern_strict(pattern):
            matches.append((self, parent, path))
        if not self.is_leaf():
            matches.extend(self.left.find_matches_strict(pattern, self, path + ['left']))
            matches.extend(self.right.find_matches_strict(pattern, self, path + ['right']))
        return matches
    
    def matches_pattern_strict(self, pattern):
        """Check if this node matches a pattern with strict value matching.
        
        Unlike matches_pattern, this function checks both:
        1. Structural match (same operators and tree shape)
        2. Value match (actual values at leaves must be equal)
        
        Args:
            pattern: ExpressionNode to match against
            
        Returns:
            True if both structure and values match exactly, False otherwise
        """
        if pattern == 'any':
            return True
        if not isinstance(pattern, ExpressionNode):
            return False
        return self._structural_match_with_values(pattern)
    
    def _structural_match_with_values(self, pattern):
        """Check structural match AND verify that actual values at leaves are equal."""
        # Check if pattern is a wildcard (all None values) - matches anything
        if pattern.value is None and pattern.operator is None and pattern.left is None and pattern.right is None:
            return True
        
        if pattern.is_leaf():
            # If pattern is a leaf, we must be a leaf AND have the same value
            if not self.is_leaf():
                return False
            return int(self.value) == int(pattern.value)  # Integer comparison
        
        if self.is_leaf():
            # If we're a leaf but pattern isn't, no match
            return False
        
        # Match operator
        if self.operator != pattern.operator:
            return False
        
        # Recursively match children with value checking
        # Handle wildcard children (None means match anything)
        left_match = True
        if pattern.left is not None:
            left_match = self.left._structural_match_with_values(pattern.left)
        
        right_match = True
        if pattern.right is not None:
            right_match = self.right._structural_match_with_values(pattern.right)
        
        return left_match and right_match
    
    def _structural_match(self, pattern):
        """Check structural match with pattern (ignoring values for variables)."""
        # Check if pattern is a wildcard (all None values) - matches anything
        if pattern.value is None and pattern.operator is None and pattern.left is None and pattern.right is None:
            return True
        
        if pattern.is_leaf():
            # If pattern is a leaf, match if we're a leaf
            return self.is_leaf()
        if self.is_leaf():
            return False
        # Match operator and recursively match children
        # Handle wildcard children (None means match anything)
        left_match = True
        if pattern.left is not None:
            left_match = self.left._structural_match(pattern.left)
        
        right_match = True
        if pattern.right is not None:
            right_match = self.right._structural_match(pattern.right)
        
        return (self.operator == pattern.operator and left_match and right_match)


class RewriteRule:
    """Represents a rewrite rule: pattern -> replacement."""
    def __init__(self, name, pattern, replacement, is_commutative=False, custom_match=None, custom_apply=None):
        self.name = name
        self.pattern = pattern  # ExpressionNode pattern
        self.replacement = replacement  # ExpressionNode or function
        self.is_commutative = is_commutative
        self.custom_match = custom_match  # Optional custom matching function
        self.custom_apply = custom_apply  # Optional custom apply function
    
    def apply(self, node, parent=None, path=None):
        """Apply this rule to a node if it matches."""
        if self.matches(node):
            # Use custom_apply if provided, otherwise use default _build_replacement
            if self.custom_apply is not None:
                return self.custom_apply(node, parent)
            return self._build_replacement(node)
        return None
    
    def matches(self, node):
        """Check if this rule matches the given node."""
        # Use custom_match if provided, otherwise use default _pattern_matches
        if self.custom_match is not None:
            return self.custom_match(node)
        return self._pattern_matches(node, self.pattern)
    
    def _pattern_matches(self, node, pattern):
        """Check if node matches pattern (simplified)."""
        if pattern == 'a + 0' or pattern == '0 + a':
            return (not node.is_leaf() and node.operator == '+' and
                   (node.left.is_leaf() and node.left.value == 0 or
                    node.right.is_leaf() and node.right.value == 0))
        elif pattern == 'a * 1' or pattern == '1 * a':
            return (not node.is_leaf() and node.operator == '*' and
                   (node.left.is_leaf() and node.left.value == 1 or
                    node.right.is_leaf() and node.right.value == 1))
        elif pattern == 'a + b':
            return not node.is_leaf() and node.operator == '+'
        elif pattern == 'a * (b + c)':
            return (not node.is_leaf() and node.operator == '*' and
                   not node.right.is_leaf() and node.right.operator == '+')
        return False
    
    def _build_replacement(self, node):
        """Build the replacement expression."""
        if self.name in ['add_zero_left', 'add_zero_right']:
            # For a + 0 -> a or 0 + a -> a
            if node.left.is_leaf() and node.left.value == 0:
                return node.right.copy()
            else:
                return node.left.copy()
        elif self.name in ['mult_one_left', 'mult_one_right']:
            # For a * 1 -> a or 1 * a -> a
            if node.left.is_leaf() and node.left.value == 1:
                return node.right.copy()
            else:
                return node.left.copy()
        elif self.replacement == 'b + a':
            # Commutativity: a + b -> b + a
            return ExpressionNode(operator='+', left=node.right.copy(), right=node.left.copy())
        elif self.replacement == '(a + b) + c':
            # Associativity: a + (b + c) -> (a + b) + c
            return ExpressionNode(
                operator='+',
                left=ExpressionNode(operator='+', left=node.left.copy(), right=node.right.left.copy()),
                right=node.right.right.copy()
            )
        elif self.replacement == 'a*b + a*c':
            # Distribution: a * (b + c) -> a*b + a*c
            return ExpressionNode(
                operator='+',
                left=ExpressionNode(operator='*', left=node.left.copy(), right=node.right.left.copy()),
                right=ExpressionNode(operator='*', left=node.left.copy(), right=node.right.right.copy())
            )
        return node.copy()


class RewritePuzzleBoard:
    """Board state for the rewrite puzzle game."""
    def __init__(self, start_expr, goal_expr, max_steps=20):
        self.current_expr = start_expr.copy() if isinstance(start_expr, ExpressionNode) else self._parse_expr(start_expr)
        # Handle goal_expr - can be a number or expression
        if isinstance(goal_expr, ExpressionNode):
            self.goal_expr = goal_expr.copy()
        elif isinstance(goal_expr, (int, float)):
            self.goal_expr = int(goal_expr)  # Convert to int
        else:
            self.goal_expr = self._parse_expr(goal_expr)
        self.max_steps = max_steps
        self.steps_taken = 0
        self.rules = self._initialize_rules()
    
    def _parse_expr(self, expr_str):
        """Parse a string expression into an ExpressionNode tree.
        
        Preserves the structure of parentheses without evaluating them.
        """
        if isinstance(expr_str, (int, float)):
            return ExpressionNode(value=int(expr_str))
        
        expr_str = str(expr_str).replace(' ', '')
        
        # Find operators outside parentheses (respecting operator precedence)
        # Addition has lower precedence, so we parse it first (rightmost + splits the expression)
        # Multiplication has higher precedence, so it's parsed in nested calls
        
        # Find the rightmost '+' at the top level (outside parentheses)
        # This will be the last operator to apply (left-associative)
        depth = 0
        add_idx = -1
        
        for i, char in enumerate(expr_str):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif depth == 0 and char == '+':  # We're at the top level
                add_idx = i  # Keep the rightmost +
        
        # Parse addition first (lower precedence - splits the expression)
        if add_idx >= 0:
            left = self._parse_expr(expr_str[:add_idx])
            right = self._parse_expr(expr_str[add_idx+1:])
            return ExpressionNode(operator='+', left=left, right=right)
        
        # Then multiplication (higher precedence - parsed in nested calls)
        depth = 0
        mult_idx = -1
        for i, char in enumerate(expr_str):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif depth == 0 and char == '*':  # We're at the top level
                mult_idx = i  # Keep the rightmost *
        
        if mult_idx >= 0:
            left = self._parse_expr(expr_str[:mult_idx])
            right = self._parse_expr(expr_str[mult_idx+1:])
            return ExpressionNode(operator='*', left=left, right=right)
        
        # Handle parentheses - remove outer parentheses if the whole expression is wrapped
        if expr_str.startswith('(') and expr_str.endswith(')'):
            # Check if parentheses wrap the entire expression
            depth = 0
            fully_wrapped = True
            for i, char in enumerate(expr_str):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    if depth == 0 and i < len(expr_str) - 1:
                        fully_wrapped = False
                        break
            
            if fully_wrapped:
                # Remove outer parentheses and parse inner
                return self._parse_expr(expr_str[1:-1])
        
        # Just a number
        try:
            val = int(float(expr_str))  # Convert to int (handles "3.0" -> 3)
            return ExpressionNode(value=val)
        except ValueError as e:
            raise ValueError(f"Could not parse expression: {expr_str}") from e
    
    def _initialize_rules(self):
        """Initialize the set of rewrite rules."""
        rules = []
        # a + 0 -> a
        # rules.append(RewriteRule("add_zero_left", "a + 0", "a"))
        # # 0 + a -> a
        # rules.append(RewriteRule("add_zero_right", "0 + a", "a"))
        # # a * 1 -> a
        rules.append(RewriteRule("mult_one_left", "a * 1", "a"))
        # 1 * a -> a
        rules.append(RewriteRule("mult_one_right", "1 * a", "a"))
        # a + b -> b + a (commutativity)
        rules.append(RewriteRule("commute_add", "a + b", "b + a", is_commutative=True))
        # a + (b + c) -> (a + b) + c (associativity)
        rules.append(RewriteRule("assoc_add", "a + (b + c)", "(a + b) + c"))
        # a * (b + c) -> a*b + a*c (distribution)
        rules.append(RewriteRule("distribute", "a * (b + c)", "a*b + a*c"))
        # Rule: evaluate addition of leaves a + b -> (value)
        rules.append(
            RewriteRule(
                "eval_add_leaves",
                "a + b",
                None,
                custom_match=lambda node: (
                    node.operator == '+' and
                    node.left is not None and node.right is not None and
                    node.left.is_leaf() and node.right.is_leaf()
                ),
                custom_apply=lambda node, _: ExpressionNode(value=int(node.left.value) + int(node.right.value))
            )
        )
        # Rule: evaluate multiplication of leaves a * b -> (value)
        rules.append(
            RewriteRule(
                "eval_mult_leaves",
                "a * b",
                None,
                custom_match=lambda node: (
                    node.operator == '*' and
                    node.left is not None and node.right is not None and
                    node.left.is_leaf() and node.right.is_leaf()
                ),
                custom_apply=lambda node, _: ExpressionNode(value=int(node.left.value) * int(node.right.value))
            )
        )
        return rules
    
    def get_all_valid_actions(self):
        """Get all valid (rule, location) pairs that can be applied."""
        valid_actions = []
        all_nodes = self.current_expr.get_all_subexpressions()
        
        for rule_idx, rule in enumerate(self.rules):
            for node in all_nodes:
                if rule.matches(node):
                    # Find the path to this node
                    path = self._find_path_to_node(self.current_expr, node)
                    action = (rule_idx, path)
                    valid_actions.append(action)
        
        return valid_actions
    
    def _find_path_to_node(self, root, target, path=[]):
        """Find the path to a target node in the tree."""
        if root == target:
            return path
        if root.is_leaf():
            return None
        left_path = self._find_path_to_node(root.left, target, path + ['left'])
        if left_path is not None:
            return left_path
        right_path = self._find_path_to_node(root.right, target, path + ['right'])
        if right_path is not None:
            return right_path
        return None
    
    def apply_action(self, rule_idx, path):
        """Apply a rewrite rule at a specific location."""
        if rule_idx >= len(self.rules):
            return False
            
        rule = self.rules[rule_idx]
        node = self._get_node_at_path(self.current_expr, path)
        
        # print(rule.matches(node))
        if node is None:
            return False
        
        if not rule.matches(node):
            return False
        
        # Use rule.apply() which handles both standard and custom rules
        replacement = rule.apply(node, parent=None, path=path)
        if replacement is None:
            return False
        
        # Replace the node - this creates a new root with the replacement
        new_expr = self._replace_node(self.current_expr, path, replacement)
        # print(new_expr)
        # Update the expression (replacement is guaranteed to be different if we got here)
        self.current_expr = new_expr
        self.steps_taken += 1
        return True
    
    def _get_node_at_path(self, root, path):
        """Get the node at a given path."""
        current = root
        for step in path:
            if current.is_leaf():
                return None
            if step == 'left':
                current = current.left
            elif step == 'right':
                current = current.right
            else:
                return None
        return current
    
    def _replace_node(self, root, path, replacement):
        """Replace a node at a given path with a replacement."""
        if not path:
            return replacement.copy()
        
        new_root = root.copy()
        current = new_root
        
        # Navigate to parent of target
        for step in path[:-1]:
            if step == 'left':
                current = current.left
            else:
                current = current.right
        
        # Replace the target
        if path[-1] == 'left':
            current.left = replacement.copy()
        else:
            current.right = replacement.copy()
        
        return new_root
    
    def is_solved(self):
        """Check if the current expression literally equals the goal expression.
        
        This function checks for literal equality (structure and values), not just
        evaluation equality. For example, "1 + 2" and "3" evaluate to the same
        value but are not literally equal.
        
        Returns:
            True if current_expr literally matches goal_expr (same structure and values),
            False otherwise
        """
        try:
            # Convert goal_expr to ExpressionNode if it's a number
            if isinstance(self.goal_expr, (int, float)):
                goal_expr_node = ExpressionNode(value=int(self.goal_expr))
            elif isinstance(self.goal_expr, ExpressionNode):
                goal_expr_node = self.goal_expr
            else:
                # If it's a string or other type, try to parse it
                goal_expr_node = self._parse_expr(self.goal_expr)
            
            # Use strict pattern matching to check literal equality
            # This checks both structure and values
            return self.current_expr.matches_pattern_strict(goal_expr_node)
        except Exception as e:
            print(e)
            return False
    
    def is_terminal(self):
        """Check if the game has ended."""
        return self.is_solved() or self.steps_taken >= self.max_steps
    
    def get_reward(self):
        """Get the reward for the current state."""
        if self.is_solved():
            return 1.0
        elif self.steps_taken >= self.max_steps:
            return 0.0
        return 0.0

