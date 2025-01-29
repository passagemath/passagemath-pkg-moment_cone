from functools import cached_property

from .representation import Representation
from .linear_group import LinearGroup

class ParticleRepresentation(Representation):
    """ Representation specific to physical particles """
    particle_cnt: int

    def __init__(self, G: LinearGroup, particle_cnt: int):
        super().__init__(G)
        self.particle_cnt = particle_cnt

    @cached_property
    def dim_cone(self) -> int:
        return self.G.rank
