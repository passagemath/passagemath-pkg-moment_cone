from functools import cached_property
from sage.all import Ring # type: ignore
from math import floor

from .utils import prod
from .typing import *

__all__ = (
    "Dimension",
)

class Dimension(tuple[int, ...]):
    """
    Dimensions of the ??? space
    
    It now contains also some useful rings used in some parts of the module.

    TODO: it may be clearer to move all this in the kind of space class
    """
    @cached_property
    def symmetries(self) -> tuple[int, ...]:
        """ Returns length of the symmetries in the dimensions """
        from .utils import group_by_block
        return tuple(length for _, length in group_by_block(self))

    @cached_property
    def sum(self) -> int:
        return sum(self)

    @cached_property
    def prod(self) -> int:
        return prod(self)
    
    @cached_property
    def Q(self) -> Ring:
        from sage.all import QQ
        return QQ
    
    @cached_property
    def QI(self) -> Ring:
        from sage.all import QQ, I
        return QQ[I]

    @cached_property
    def QZ(self) -> Ring:
        from sage.all import QQ
        return QQ["z"]
    
    @cached_property
    def QV(self) -> Ring:
        from .polynomial_ring import variable_name
        from .weight import Weight
        from sage.all import QQ, PolynomialRing
        variables_names = [variable_name(chi) for chi in Weight.all(self)]
        return PolynomialRing(QQ, variables_names)
    
    @cached_property
    def QIV(self) -> Ring:
        from .polynomial_ring import variable_name
        from .weight import Weight
        from sage.all import QQ, I, PolynomialRing
        from itertools import chain
        variables_names = chain.from_iterable(
            (variable_name(chi, seed="vr"), variable_name(chi, seed="vi"))
            for chi in Weight.all(self)
        )
        return PolynomialRing(QQ[I], list(variables_names))

    def uMAX(self,e: "Dimension")->int: # Maximal value of u obtained by extending a e-1-PS to a d-1-PS
        """
        For a dimension vector d=(d_i), and a list of number of Levi blocks in each d_i,
        computes the maximal dimension of a nilradical.
        """
        return(sum([floor(self[i]*self[i]/2*(1-1/e[i])) for i in range(len(self))]))
        
        
