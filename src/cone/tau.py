from .typing import *
from .matrix import PartialMatrix
from .weight import *

class Tau(Generic[Integer]):
    """ Tuple of partition along with a coefficient """
    __slots__ = 'ccomponent', '_components'
    ccomponent: Optional[Integer]
    _components: PartialMatrix[Integer]

    def __init__(self, components: Sequence[Sequence[Integer]] | PartialMatrix[Integer], ccomponent: Optional[Integer] = None):
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

    @property
    def d(self) -> tuple[int, ...]:
        """ Length of each component """
        return self._components.sizes
    
    @property
    def components(self) -> Sequence[Sequence[Integer]]:
        """ Sequence of the components of tau """
        return self._components.columns

    @property
    def reduced(self) -> "ReducedTau[Integer]":
        """ Returns reduced form of tau """
        return ReducedTau(self)
    
    def __repr__(self) -> str:
        return f"{self.ccomponent} | " + " | ".join(" ".join(map(str, c)) for c in self.components)
    
    def dotV(self, weight: WeightV, ccomponent: Optional[Integer] = None) -> Integer:
        """ Scalar product of tau with a weight of V """
        # TODO : rename to dot_weight
        assert not (self.ccomponent is None and ccomponent is None)
        if ccomponent is None:
            ccomponent = self.ccomponent
        return ccomponent + cast(Integer, sum(c[wi] for c, wi in zip(self.components, weight))) # type: ignore
    
    def dotU(self, weight: WeightU) -> Integer:
        """ Scalar product of tau with a weight of U """
        # TODO : rename to dot_root
        k, i, j = weight
        c = self.components[k]
        return c[i] - c[j] # type: ignore
    
    @property
    def is_regular(self) -> bool:
        """ Check is tau is regular assuming it is dominant """
        return not any(any(a == b for a, b in itertools.pairwise(c)) for c in self.components)
    
    def positive_weights(self, weights: Iterable[WeightV]) -> dict[Integer, list[WeightV]]:
        """ Inverse image of each non-zero p = <w, tau> for each w in weights """
        result = {}
        for chi in weights:
            p = self.dotV(chi)
            if p >= 0:
                result.setdefault(p, []).append(chi)
        return result


class ReducedTau(Generic[Integer]):
    """ Tau in a reduction form """
    __slots__ = 'ccomponent', 'values', 'mult'
    ccomponent: Optional[Integer]
    values: PartialMatrix[Integer]
    mult: PartialMatrix[Integer]

    def __init__(self, tau: Tau[Integer]):
        from .utils import compress
        values: PartialMatrix[Integer] = PartialMatrix(max(tau.d), len(tau))
        mult: PartialMatrix[Integer] = PartialMatrix(max(tau.d), len(tau))

        for j, component in enumerate(tau.components):
            for v, m in compress(component):
                values.append(j, v)
                mult.append(j, cast(Integer, m))

        self.ccomponent = tau.ccomponent
        self.values = values
        self.mult = mult

    def __len__(self) -> int:
        return self.values.shape[1]
    
    @property
    def small_d(self) -> tuple[int, ...]:
        return self.values.sizes
    
    def __getitem__(self, idx: tuple[int, int]) -> tuple[Integer, Integer]:
        return self.values[idx], self.mult[idx]
    
    def __repr__(self) -> str:
        return f"{self.ccomponent} | " + " | ".join(
            " ".join(f"{v}^{m}" for v, m in zip(cv, cm))
            for cv, cm in zip(self.values.columns, self.mult.columns)
        )
    
    def Pzero(self, n: Integer) -> Iterable[WeightV]:
        """ Search for weights w of V so that n + sum_k tau_red[w_k, k] = 0 """
        for weight in all_weights_V(self.small_d):
            s = cast(Integer, sum(self.values[i, j] for j, i in enumerate(weight)))
            if s + n == 0:
                yield weight