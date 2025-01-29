from functools import cached_property
import itertools

from .typing import *
from .particle import ParticleRepresentation
from .weight import Weight
from .linear_group import LinearGroup

class FermionWeight(Weight):
    def __init__(self, G: LinearGroup, **kwargs):
        super().__init__(G, **kwargs)
        self.__weights = tuple(weights)
    
    @staticmethod
    def all(G: LinearGroup) -> Iterable[Weight]:
        for i,w in enumerate(itertools.combinations(range(self.G[0]),self.nb_part)):
            print('w',[list(w)])
            yield Weight(self.G,as_list_of_list=[list(w)],index=i)


class FermionRepresentration(ParticleRepresentation):
    @cached_property
    def Weight(self) -> type[FermionWeight]:
        return FermionWeight

    @cached_property
    def dim(self) -> int:
        from math import comb
        return comb(self.G.rank, self.particle_cnt)
