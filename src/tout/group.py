from functools import cached_property
from typing import Generic,TypeVar
T=TypeVar('T')
from collections.abc import Iterable, Sequence, Callable

from utils import prod, symmetries
#from .typing import *


__all__ = (
    "LinGroup",
)

class LinGroup(tuple[int, ...]):
    """
    Product of Linear Groups GL(d_i)

    Examples:
    >>> G = LinGroup((4, 4, 3, 2))
    >>> G
    GL(4)xGL(4)xGL(3)xGL(2)
    >>> G.rank
    13
    >>> G.dim
    45
    >>> G.dimU
    20
    >>> G.outer
    (1,3)
    It should also be noted that this class ensure uniqueness of an instance
    for a given sequence of dimensions:
    >>> G2=LinGroup([1,4,4,4])
    >>> G==G2
    True
    """
    all_instances: dict["LinGroup", "LinGroup"] = {}

    def __new__(cls, dimensions):
        """ Construction with reusing of already computed Dimension instance """
        d = super().__new__(cls, dimensions)
        return cls.all_instances.setdefault(d, d)

    def __repr__(self) -> str:
        return 'x'.join(f'GL({i})' for i in self)

    @cached_property
    def outer(self) -> tuple[int, ...]:
        """ Returns length of the symmetries in the dimensions """
        return tuple(symmetries(self))

    @cached_property
    def rank(self) -> int:
        return sum(self)

    @cached_property
    def dim(self) -> int:
        """ Rank of the group G """
        return sum(i**2 for i in self)

    @cached_property
    def dimU(self) -> int:
        """ Dimension of the unipotent subgroup U """
        g = sum(i**2 for i in self)
        return (self.dim - self.rank) // 2

        

