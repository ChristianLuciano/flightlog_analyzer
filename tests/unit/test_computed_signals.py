"""Tests for computed signals functionality."""

import pytest
import numpy as np
import pandas as pd

from src.computed_signals.parser import FormulaParser
from src.computed_signals.functions import BUILTIN_FUNCTIONS
from src.computed_signals.dependencies import DependencyResolver
from src.core.exceptions import FormulaParseError, FormulaSyntaxError


class TestFormulaParser:
    """Tests for FormulaParser class."""

    def test_parse_simple_expression(self):
        """Test parsing simple expressions."""
        parser = FormulaParser()

        assert parser.validate('x + y')
        assert parser.validate('x * 2 + y / 3')
        assert parser.validate('sqrt(x**2 + y**2)')

    def test_evaluate_arithmetic(self):
        """Test evaluating arithmetic expressions."""
        parser = FormulaParser()

        result = parser.evaluate('x + y', {'x': 1, 'y': 2})
        assert result == 3

        result = parser.evaluate('x * y', {'x': np.array([1, 2, 3]), 'y': 2})
        assert np.array_equal(result, np.array([2, 4, 6]))

    def test_evaluate_functions(self):
        """Test evaluating function calls."""
        parser = FormulaParser()

        result = parser.evaluate('sqrt(x)', {'x': 4})
        assert result == 2.0

        result = parser.evaluate('sin(x)', {'x': 0})
        assert result == 0.0

    def test_evaluate_conditional(self):
        """Test evaluating conditional expressions."""
        parser = FormulaParser()

        x = np.array([1, -2, 3, -4])
        result = parser.evaluate('where(x > 0, x, 0)', {'x': x})
        assert np.array_equal(result, np.array([1, 0, 3, 0]))

    def test_extract_variables(self):
        """Test extracting variables from formula."""
        parser = FormulaParser()

        vars = parser.extract_variables('x + y * z')
        assert vars == {'x', 'y', 'z'}

        vars = parser.extract_variables('sqrt(a**2 + b**2)')
        assert vars == {'a', 'b'}

    def test_invalid_syntax_error(self):
        """Test error on invalid syntax."""
        parser = FormulaParser()

        with pytest.raises(FormulaSyntaxError):
            parser.validate('x + ')

    def test_unknown_variable_error(self):
        """Test error on unknown variable."""
        parser = FormulaParser()

        with pytest.raises(FormulaParseError):
            parser.evaluate('x + unknown', {'x': 1})


class TestBuiltinFunctions:
    """Tests for built-in functions."""

    def test_math_functions(self):
        """Test mathematical functions."""
        assert BUILTIN_FUNCTIONS['sqrt'](4) == 2.0
        assert np.isclose(BUILTIN_FUNCTIONS['sin'](np.pi / 2), 1.0)
        assert np.isclose(BUILTIN_FUNCTIONS['cos'](0), 1.0)

    def test_signal_functions(self):
        """Test signal processing functions."""
        x = np.array([1, 2, 3, 4, 5])

        diff_result = BUILTIN_FUNCTIONS['diff'](x)
        assert diff_result[1] == 1  # First difference

        cumsum_result = BUILTIN_FUNCTIONS['cumsum'](x)
        assert cumsum_result[-1] == 15

    def test_statistical_functions(self):
        """Test statistical functions."""
        x = np.array([1, 2, 3, 4, 5])

        assert BUILTIN_FUNCTIONS['mean'](x) == 3.0
        assert BUILTIN_FUNCTIONS['min'](x) == 1
        assert BUILTIN_FUNCTIONS['max'](x) == 5

    def test_geographic_functions(self):
        """Test geographic functions."""
        # Distance from same point should be 0
        dist = BUILTIN_FUNCTIONS['haversine'](0, 0, 0, 0)
        assert dist == 0

        # Known distance test
        dist = BUILTIN_FUNCTIONS['haversine'](0, 0, 0, 1)
        assert 100000 < dist < 120000  # ~111km


class TestDependencyResolver:
    """Tests for DependencyResolver class."""

    def test_add_dependency(self):
        """Test adding dependencies."""
        resolver = DependencyResolver()

        resolver.add_dependency('c', 'a')
        resolver.add_dependency('c', 'b')

        deps = resolver.get_dependencies('c')
        assert 'a' in deps
        assert 'b' in deps

    def test_cycle_detection(self):
        """Test circular dependency detection."""
        resolver = DependencyResolver()

        resolver.add_dependency('b', 'a')
        resolver.add_dependency('c', 'b')

        assert not resolver.has_cycle()

        resolver.add_dependency('a', 'c')
        assert resolver.has_cycle()

    def test_computation_order(self):
        """Test topological ordering."""
        resolver = DependencyResolver()

        resolver.add_dependency('c', 'a')
        resolver.add_dependency('c', 'b')
        resolver.add_dependency('d', 'c')

        order = resolver.get_computation_order()
        # a and b should come before c, c before d
        assert order.index('a') < order.index('c')
        assert order.index('b') < order.index('c')

    def test_affected_signals(self):
        """Test affected signals calculation."""
        resolver = DependencyResolver()

        resolver.add_dependency('b', 'a')
        resolver.add_dependency('c', 'a')
        resolver.add_dependency('d', 'b')

        affected = resolver.get_affected_signals('a')
        assert 'b' in affected
        assert 'c' in affected
        assert 'd' in affected

