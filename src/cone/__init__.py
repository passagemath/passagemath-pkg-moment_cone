from sage.all import SymmetricFunctions, QQ, Integer as SageInteger
sym_f = SymmetricFunctions(QQ).s()
#import sage.libs.lrcalc.lrcalc as lr
import itertools
from dataclasses import dataclass


from .utils import *
from .typing import *

from .matrix import PartialMatrix
from .partition import Partition

    

def fct_Weights_of_Nu(nu: PartialMatrix[Partition[Integer]]) -> PartialMatrix[Integer]:
    """ Constructing size_nu from nu """
    size_nu = PartialMatrix[Integer](*nu.shape)

    for j, col in enumerate(nu.columns):
        size_nu.extend(j, (cast(Integer, sum(p)) for p in col))

    return size_nu



##### Pour 444, il y a un cas où on rentre dans LR_multi avec L=[] (liste vide) et nu=[].
## Cela bugue mais je ne comprends pas si c'est normal d'avoir une telle entrée
