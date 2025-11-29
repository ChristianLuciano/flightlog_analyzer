"""
Dependency resolution for computed signals.

Handles dependency graph management and circular dependency detection.
"""

from typing import Dict, Set, List, Optional
from collections import defaultdict


class DependencyResolver:
    """
    Manages signal dependencies and detects cycles.

    Uses topological sorting to determine computation order.
    """

    def __init__(self):
        """Initialize dependency resolver."""
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._dependents: Dict[str, Set[str]] = defaultdict(set)

    def add_dependency(self, signal: str, depends_on: str) -> None:
        """
        Add a dependency relationship.

        Args:
            signal: The signal that depends on another.
            depends_on: The signal it depends on.
        """
        self._dependencies[signal].add(depends_on)
        self._dependents[depends_on].add(signal)

    def remove_dependency(self, signal: str, depends_on: str) -> None:
        """Remove a dependency relationship."""
        self._dependencies[signal].discard(depends_on)
        self._dependents[depends_on].discard(signal)

    def remove_node(self, signal: str) -> None:
        """Remove a signal and all its dependencies."""
        # Remove from dependencies
        for dep in list(self._dependencies[signal]):
            self._dependents[dep].discard(signal)
        del self._dependencies[signal]

        # Remove from dependents
        for dependent in list(self._dependents[signal]):
            self._dependencies[dependent].discard(signal)
        del self._dependents[signal]

    def get_dependencies(self, signal: str) -> Set[str]:
        """Get direct dependencies of a signal."""
        return self._dependencies.get(signal, set()).copy()

    def get_all_dependencies(self, signal: str) -> Set[str]:
        """Get all dependencies (transitive closure)."""
        all_deps = set()
        to_process = list(self._dependencies.get(signal, set()))

        while to_process:
            dep = to_process.pop()
            if dep not in all_deps:
                all_deps.add(dep)
                to_process.extend(self._dependencies.get(dep, set()))

        return all_deps

    def get_dependents(self, signal: str) -> Set[str]:
        """Get signals that depend on this signal."""
        return self._dependents.get(signal, set()).copy()

    def has_cycle(self) -> bool:
        """Check if the dependency graph has cycles."""
        visited = set()
        rec_stack = set()

        def has_cycle_util(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for dep in self._dependencies.get(node, set()):
                if dep not in visited:
                    if has_cycle_util(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in self._dependencies:
            if node not in visited:
                if has_cycle_util(node):
                    return True

        return False

    def get_computation_order(self) -> List[str]:
        """
        Get topological order for computation.

        Returns:
            List of signal names in computation order.
        """
        # Collect all nodes (both those with dependencies and those depended upon)
        all_nodes = set(self._dependencies.keys()) | set(self._dependents.keys())

        in_degree = defaultdict(int)
        for signal in self._dependencies:
            for dep in self._dependencies[signal]:
                in_degree[signal] += 1

        # Start with signals that have no dependencies (in_degree == 0)
        queue = [s for s in all_nodes if in_degree[s] == 0]
        order = []

        while queue:
            signal = queue.pop(0)
            order.append(signal)

            for dependent in self._dependents.get(signal, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return order

    def get_affected_signals(self, signal: str) -> Set[str]:
        """Get all signals affected if this signal changes."""
        affected = set()
        to_process = list(self._dependents.get(signal, set()))

        while to_process:
            dependent = to_process.pop()
            if dependent not in affected:
                affected.add(dependent)
                to_process.extend(self._dependents.get(dependent, set()))

        return affected

    def clear(self) -> None:
        """Clear all dependencies."""
        self._dependencies.clear()
        self._dependents.clear()

