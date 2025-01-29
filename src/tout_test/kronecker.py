__all__ = (
    'KroneckerWeight',
    'KroneckerRepresentation',
)

from functools import cached_property
import itertools

from .typing import *
from .representation import Representation
from .weight import Weight
from .linear_group import LinearGroup
from .rings import Vector, vector

"""
    # Move to representation
    @staticmethod
    def all(G: LinearGroup) -> Iterable[Weight]:
        " Returns all possible weights for a given sequence of dimensions, in the lexicographical order "
        for idx, w in enumerate(itertools.product(*(range(di) for di in G))):
            yield KroneckerWeight(G, w, index=idx)

    # Move to representation
    @cached_property
    def index(self) -> int:
        from operator import mul
        stride = itertools.accumulate(reversed(self.G[1:]), mul, initial=1)
        return sum(v * s for v, s in zip(reversed(self.__weights), stride))
"""

class KroneckerWeight(Weight):
    """
    Weight class for the Kronecker representation
    """
    __weights: tuple[int, ...]

    def __init__(self, G: LinearGroup, weights: Iterable[int], **kwargs):
        super().__init__(G, **kwargs)
        self.__weights = tuple(weights)

    @cached_property
    def as_vector(self) -> Vector:
        from .rings import ZZ
        v = vector(ZZ, self.G.rank)
        for shift, x in zip(itertools.accumulate(self.G, initial=0), self.__weights):
            v[shift + x] = 1
        return v

    def __iter__(self) -> Iterator[int]:
        return iter(self.__weights)

    def __le__(self, other: Weight) -> bool:
        """ Implementation of self <= other (partial ordering)"""
        if not isinstance(other, KroneckerWeight):
            return NotImplemented
        return all(ws >= wo for ws, wo in zip(self, other))
    
    def leq(self,
            other: "Weight",
            symmetries: Optional[Iterable[int]] = None) -> bool:
        if symmetries is None:
            return self <= other
        else:
            return super().leq(other, symmetries)

class KroneckerRepresentation(Representation):
    Weight = KroneckerWeight

    #@cached_property
    #def Weight(self) -> type[KroneckerWeight]:
    #    return KroneckerWeight
    
    @cached_property
    def dim_cone(self) -> int:
        return self.G.rank - len(self.G) + 1
    
    @cached_property
    def dim(self) -> int:
        from .utils import prod
        return prod(self.G)

