from .typing import *
from .matrix import PartialMatrix
from .weight import Weight
from .root import Root

import itertools
from functools import cached_property

class Tau:
    """ Tuple of partition along with a coefficient """
    __slots__ = 'ccomponent', '_components'
    ccomponent: Optional[int]
    _components: PartialMatrix[int]

    def __init__(self, components: Sequence[Sequence[int]] | PartialMatrix[int], ccomponent: Optional[int] = None):
        """ Tau initialization from a sequence of sub-group or directly from a partial matrix """
        self.ccomponent = ccomponent
        if isinstance(components, PartialMatrix):
            self._components = components # FIXME: copy ?
        else:
            self._components = PartialMatrix(max(len(c) for c in components), len(components))
            for j, c in enumerate(components):
                self._components.extend(j, c)
    
    def __len__(self) -> int:
        """ Number of components """
        return self._components.shape[1]

    @cached_property
    def d(self) -> tuple[int, ...]:
        """ Length of each component """
        return self._components.sizes
    
    @cached_property
    def components(self) -> Sequence[Sequence[int]]:
        """ Sequence of the components of tau """
        return self._components.columns

    @cached_property
    def reduced(self) -> "ReducedTau":
        """ Returns reduced form of tau """
        return ReducedTau(self)
    
    def __repr__(self) -> str:
        return f"{self.ccomponent} | " + " | ".join(" ".join(map(str, c)) for c in self.components)
    
    def dot_weight(self, weight: Weight, ccomponent: Optional[int] = None) -> int:
        """ Scalar product of tau with a weight of V """
        assert not (self.ccomponent is None and ccomponent is None)
        if ccomponent is None:
            ccomponent = self.ccomponent
        return cast(int, ccomponent) + sum(c[wi] for c, wi in zip(self.components, weight))
    
    def dot_root(self, root: Root) -> int:
        """ Scalar product of tau with a weight of U """
        c = self.components[root.k]
        return c[root.i] - c[root.j]
    
    @cached_property
    def is_regular(self) -> bool:
        """ Check is tau is regular assuming it is dominant """
        return not any(any(a == b for a, b in itertools.pairwise(c)) for c in self.components)
    
    def positive_weights(self, weights: Iterable[Weight]) -> dict[int, list[Weight]]:
        """ Inverse image of each non-zero p = <w, tau> for each w in weights """
        result = {}
        for chi in weights:
            p = self.dot_weight(chi)
            if p >= 0:
                result.setdefault(p, []).append(chi)
        return result


class ReducedTau:
    """ Tau in a reduction form """
    __slots__ = 'ccomponent', 'values', 'mult'
    ccomponent: Optional[int]
    values: PartialMatrix[int]
    mult: PartialMatrix[int]

    def __init__(self, tau: Tau):
        from .utils import compress
        values: PartialMatrix[int] = PartialMatrix(max(tau.d), len(tau))
        mult: PartialMatrix[int] = PartialMatrix(max(tau.d), len(tau))

        for j, component in enumerate(tau.components):
            for v, m in compress(component):
                values.append(j, v)
                mult.append(j, cast(int, m))

        self.ccomponent = tau.ccomponent
        self.values = values
        self.mult = mult

    def __len__(self) -> int:
        return self.values.shape[1]
    
    @cached_property
    def small_d(self) -> tuple[int, ...]:
        return self.values.sizes
    
    def __getitem__(self, idx: tuple[int, int]) -> tuple[int, int]:
        return self.values[idx], self.mult[idx]
    
    def __repr__(self) -> str:
        return f"{self.ccomponent} | " + " | ".join(
            " ".join(f"{v}^{m}" for v, m in zip(cv, cm))
            for cv, cm in zip(self.values.columns, self.mult.columns)
        )
    
    @cached_property
    def Pzero(self) -> Iterable[Weight]:
        """ Search for weights w of V so that C_component + sum_k tau_red[w_k, k] = 0 """
        as_tau = Tau(self.values, self.ccomponent)
        for weight in Weight.all(self.small_d):
            if as_tau.dot_weight(weight) == 0:
                yield weight