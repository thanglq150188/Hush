from typing import TypeVar, Generic, Dict, DefaultDict, Set
from collections import defaultdict

K = TypeVar('K')
V = TypeVar('V')


class BiMapReverse(Generic[V, K]):
    def __init__(self, parent: 'BiMap[K, V]'):
        self._parent = parent

    def __getitem__(self, value: V) -> Set[K]:
        """Only V -> Set[K] lookups"""
        return self._parent._reverse.get(value, set())

    def __contains__(self, value: V) -> bool:
        return value in self._parent._reverse


class BiMap(Generic[K, V]):
    def __init__(self):
        self.forward: Dict[K, V] = {}
        self._reverse: DefaultDict[V, Set[K]] = defaultdict(set)
        self._reverse_view = BiMapReverse(self)

    @property
    def reverse(self) -> BiMapReverse[V, K]:
        return self._reverse_view

    def __getitem__(self, key: K) -> V:
        """Only K -> V lookups"""
        if key not in self.forward:
            raise KeyError(f"Key '{key}' not found")
        return self.forward[key]

    def __setitem__(self, key: K, value: V):
        # Remove old mapping if exists
        if key in self.forward:
            old_value = self.forward[key]
            self._reverse[old_value].discard(key)
            if not self._reverse[old_value]:
                del self._reverse[old_value]

        # Set new mapping
        self.forward[key] = value
        self._reverse[value].add(key)

    def __contains__(self, key: K) -> bool:
        return key in self.forward

    def __repr__(self) -> str:
        forward = dict(self.forward)
        reverse = {v: set(ks) for v, ks in self._reverse.items()}
        return f"BiMap(forward={forward}, reverse={reverse})"
