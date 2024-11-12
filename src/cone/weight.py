from .typing import *
import itertools

WeightV: TypeAlias = Iterable[int]
WeightU: TypeAlias = tuple[int, int, int]

def all_weights_V(d: Iterable[int]) -> Iterable[WeightV]:
    """ Returns all possible weights pf V for a given sequence of length """
    return itertools.product(*(range(di) for di in d))

def all_weights_U(d: Iterable[int]) -> Iterable[WeightU]:
    """ Returns all possible weights pf U for a given sequence of length """
    raise NotImplemented