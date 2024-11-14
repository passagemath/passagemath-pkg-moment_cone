from .typing import *
import itertools
import operator
from functools import cached_property

"""
TODO:
- weight of V as a class, natural numbers
- <= operator
- index_of method
- OrbitWeightSd method (needs multiset_permutations)

Remarque :
- WeightV -> Weight
- WeightU -> Root
"""

class Weight(Sequence[int]):
    """ A weight on which tau can be applied """
    __slots__ = '_weights', 'index'
    _weights: tuple[int, ...]
    index: Optional[int] # Index in the lexicographical order

    def __init__(self, weights: Iterable[int], index: Optional[int] = None):
        self._weights = tuple(weights)
        self.index = index

    @staticmethod
    def all(d: Iterable[int]) -> Iterable["Weight"]:
        """ Returns all possible weights for a given sequence of dimensions, in the lexicographical order """
        for idx, w in enumerate(itertools.product(*(range(di) for di in d))):
            yield Weight(w, idx)

    def index_in(self, d: Sequence[int]) -> int:
        """ Returns index of this weight in the lexicographical order for given dimensions (see `all` method)"""
        stride = itertools.accumulate(reversed(d[1:]), operator.mul, initial=1)
        return sum(v * s for v, s in zip(reversed(self._weights), stride))
    
    def __len__(self) -> int:
        return len(self._weights)
    
    def __getitem__(self, idx: int) -> int:
        return self._weights[idx]
    
    def __iter__(self) -> Iterator[int]:
        return iter(self._weights)


def all_weights_U(d: Iterable[int]) -> Iterable[WeightU]:
    """ Returns all possible weights of U for a given sequence of length """
    # TODO: verify that the actual definition of this weight are so that i < j
    for k, l in enumerate(d):
        for i, j in itertools.combinations(range(l), 2):
            yield k, i, j

