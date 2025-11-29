"""
Computed signal calculation engine.

Handles formula evaluation, caching, and dependency management
for derived signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Callable
import logging

from .parser import FormulaParser
from .cache import ComputedSignalCache
from .dependencies import DependencyResolver
from ..core.types import ComputedSignalConfig, SignalPath
from ..core.exceptions import (
    ComputedSignalError, CircularDependencyError,
    SignalNotFoundError, ComputationError
)
from ..core.constants import MAX_COMPUTATION_TIME_SEC

logger = logging.getLogger(__name__)


class ComputedSignalEngine:
    """
    Engine for computing derived signals from formulas.

    Supports mathematical expressions, custom functions,
    and automatic dependency resolution.
    """

    def __init__(self, data_provider):
        """
        Initialize ComputedSignalEngine.

        Args:
            data_provider: Data provider for accessing base signals.
        """
        self.data_provider = data_provider
        self.parser = FormulaParser()
        self.cache = ComputedSignalCache()
        self.dependency_resolver = DependencyResolver()
        self._definitions: Dict[str, ComputedSignalConfig] = {}

    def register_signal(self, name: str, config: ComputedSignalConfig) -> None:
        """
        Register a computed signal definition.

        Args:
            name: Signal name.
            config: Signal configuration.
        """
        # Validate formula
        self.parser.validate(config.get("formula", ""))

        # Check for circular dependencies
        inputs = config.get("inputs", [])
        for input_path in inputs:
            if input_path in self._definitions:
                self.dependency_resolver.add_dependency(name, input_path)

        if self.dependency_resolver.has_cycle():
            raise CircularDependencyError(
                f"Adding signal '{name}' would create circular dependency"
            )

        self._definitions[name] = config
        logger.info(f"Registered computed signal: {name}")

    def compute(
        self,
        name: str,
        time_range: Optional[tuple] = None,
        force_recompute: bool = False
    ) -> pd.Series:
        """
        Compute a signal by name.

        Args:
            name: Signal name.
            time_range: Optional (start, end) time range.
            force_recompute: Force recomputation ignoring cache.

        Returns:
            Computed signal as pandas Series.
        """
        if name not in self._definitions:
            raise SignalNotFoundError(f"Computed signal not found: {name}")

        # Check cache
        cache_key = f"{name}_{time_range}"
        if not force_recompute:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        config = self._definitions[name]
        formula = config.get("formula", "")
        inputs = config.get("inputs", [])
        parameters = config.get("parameters", {})

        # Gather input data
        input_data = {}
        for input_path in inputs:
            try:
                # Check if it's another computed signal
                if input_path in self._definitions:
                    input_data[input_path] = self.compute(input_path, time_range)
                else:
                    ts, values = self.data_provider.get_signal_with_timestamp(input_path)
                    input_data[input_path] = values
            except Exception as e:
                raise SignalNotFoundError(f"Input signal not found: {input_path}")

        # Add parameters to context
        context = {**input_data, **parameters}

        # Evaluate formula
        try:
            result = self.parser.evaluate(formula, context)
            result = pd.Series(result)
        except Exception as e:
            raise ComputationError(f"Error computing '{name}': {e}")

        # Cache result
        self.cache.set(cache_key, result)

        return result

    def unregister_signal(self, name: str) -> bool:
        """Unregister a computed signal."""
        if name in self._definitions:
            del self._definitions[name]
            self.dependency_resolver.remove_node(name)
            self.cache.invalidate_pattern(f"{name}_*")
            return True
        return False

    def list_signals(self) -> List[str]:
        """List all registered computed signals."""
        return list(self._definitions.keys())

    def get_definition(self, name: str) -> Optional[ComputedSignalConfig]:
        """Get signal definition."""
        return self._definitions.get(name)

    def get_dependencies(self, name: str) -> List[str]:
        """Get dependencies for a signal."""
        if name not in self._definitions:
            return []
        return self._definitions[name].get("inputs", [])

    def invalidate_cache(self, name: Optional[str] = None) -> None:
        """Invalidate cache for signal or all signals."""
        if name:
            self.cache.invalidate_pattern(f"{name}_*")
        else:
            self.cache.clear()

    def validate_formula(self, formula: str) -> tuple:
        """
        Validate a formula.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            self.parser.validate(formula)
            return True, None
        except Exception as e:
            return False, str(e)

