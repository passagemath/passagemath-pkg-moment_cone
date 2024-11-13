from .typing import *
import itertools

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

WeightV: TypeAlias = Iterable[int]
WeightU: TypeAlias = tuple[int, int, int]

def all_weights_V(d: Iterable[int]) -> Iterable[WeightV]:
    """ Returns all possible weights of V for a given sequence of length """
    return itertools.product(*(range(di) for di in d))

def all_weights_U(d: Iterable[int]) -> Iterable[WeightU]:
    """ Returns all possible weights of U for a given sequence of length """
    # TODO: verify that the actual definition of this weight are so that i < j
    for k, l in enumerate(d):
        for i, j in itertools.combinations(range(l), 2):
            yield k, i, j

