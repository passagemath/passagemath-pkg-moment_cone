from functools import cached_property

from .particle import ParticleRepresentation

class BosonRepresentration(ParticleRepresentation):
    @cached_property
    def dim(self) -> int:
        from math import comb
        return comb(
            self.G.rank + self.particle_cnt - 1,
            self.G.rank - 1
        )