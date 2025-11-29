"""
Formula parser for computed signals.

Provides safe parsing and evaluation of mathematical expressions
with support for signal references and built-in functions.
"""

import ast
import numpy as np
from typing import Dict, Any, Set, Optional
import operator

from .functions import BUILTIN_FUNCTIONS
from ..core.exceptions import FormulaParseError, FormulaSyntaxError


# Safe operators
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

COMPARISON_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

BOOL_OPS = {
    ast.And: np.logical_and,
    ast.Or: np.logical_or,
}


class FormulaParser:
    """
    Safe formula parser and evaluator.

    Uses AST parsing to safely evaluate mathematical expressions
    without executing arbitrary code.
    """

    def __init__(self):
        """Initialize FormulaParser."""
        self._functions = BUILTIN_FUNCTIONS.copy()
        self._constants = {
            "pi": np.pi,
            "e": np.e,
            "nan": np.nan,
            "inf": np.inf,
        }

    def add_function(self, name: str, func: callable) -> None:
        """Add a custom function."""
        self._functions[name] = func

    def validate(self, formula: str) -> bool:
        """
        Validate formula syntax.

        Args:
            formula: Formula string.

        Returns:
            True if valid.

        Raises:
            FormulaSyntaxError: If syntax is invalid.
        """
        try:
            tree = ast.parse(formula, mode='eval')
            self._validate_node(tree.body)
            return True
        except SyntaxError as e:
            raise FormulaSyntaxError(f"Syntax error: {e}")
        except Exception as e:
            raise FormulaParseError(f"Parse error: {e}")

    def _validate_node(self, node: ast.AST) -> None:
        """Recursively validate AST node."""
        # Node types for expressions (ast.Constant replaces deprecated ast.Num/ast.Str)
        allowed_node_types = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare,
            ast.BoolOp, ast.IfExp, ast.Call, ast.Name, ast.Constant,
            ast.Attribute, ast.Subscript, ast.Load, ast.Tuple, ast.List,
            ast.keyword,
        )

        # Operator types (children of BinOp, UnaryOp, Compare, BoolOp)
        allowed_operator_types = (
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
            ast.USub, ast.UAdd, ast.Not,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.And, ast.Or,
        )

        allowed_types = allowed_node_types + allowed_operator_types

        if not isinstance(node, allowed_types):
            raise FormulaParseError(f"Disallowed node type: {type(node).__name__}")

        for child in ast.iter_child_nodes(node):
            self._validate_node(child)

    def evaluate(self, formula: str, context: Dict[str, Any]) -> Any:
        """
        Evaluate formula with given context.

        Args:
            formula: Formula string.
            context: Variable name to value mapping.

        Returns:
            Evaluation result.
        """
        tree = ast.parse(formula, mode='eval')
        return self._eval_node(tree.body, context)

    def _eval_node(self, node: ast.AST, context: Dict[str, Any]) -> Any:
        """Recursively evaluate AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            name = node.id
            if name in context:
                return context[name]
            elif name in self._constants:
                return self._constants[name]
            elif name in self._functions:
                return self._functions[name]
            else:
                raise FormulaParseError(f"Unknown variable: {name}")

        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, context)
            right = self._eval_node(node.right, context)
            op = SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise FormulaParseError(f"Unsupported operator: {type(node.op)}")
            return op(left, right)

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, context)
            op = SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise FormulaParseError(f"Unsupported operator: {type(node.op)}")
            return op(operand)

        elif isinstance(node, ast.Compare):
            left = self._eval_node(node.left, context)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, context)
                op_func = COMPARISON_OPS.get(type(op))
                if op_func is None:
                    raise FormulaParseError(f"Unsupported comparison: {type(op)}")
                result = op_func(left, right)
                left = right
            return result

        elif isinstance(node, ast.BoolOp):
            values = [self._eval_node(v, context) for v in node.values]
            op_func = BOOL_OPS.get(type(node.op))
            result = values[0]
            for v in values[1:]:
                result = op_func(result, v)
            return result

        elif isinstance(node, ast.IfExp):
            test = self._eval_node(node.test, context)
            if isinstance(test, np.ndarray):
                return np.where(
                    test,
                    self._eval_node(node.body, context),
                    self._eval_node(node.orelse, context)
                )
            return self._eval_node(node.body if test else node.orelse, context)

        elif isinstance(node, ast.Call):
            func = self._eval_node(node.func, context)
            args = [self._eval_node(arg, context) for arg in node.args]
            kwargs = {kw.arg: self._eval_node(kw.value, context) for kw in node.keywords}
            return func(*args, **kwargs)

        elif isinstance(node, ast.Attribute):
            value = self._eval_node(node.value, context)
            return getattr(value, node.attr)

        else:
            raise FormulaParseError(f"Unsupported node: {type(node)}")

    def extract_variables(self, formula: str) -> Set[str]:
        """Extract variable names from formula."""
        tree = ast.parse(formula, mode='eval')
        variables = set()
        self._collect_names(tree.body, variables)
        # Remove functions and constants
        variables -= set(self._functions.keys())
        variables -= set(self._constants.keys())
        return variables

    def _collect_names(self, node: ast.AST, names: Set[str]) -> None:
        """Collect all Name nodes."""
        if isinstance(node, ast.Name):
            names.add(node.id)
        for child in ast.iter_child_nodes(node):
            self._collect_names(child, names)

