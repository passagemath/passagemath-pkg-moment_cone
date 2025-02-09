import itertools

from numpy.typing import NDArray

from .typing import *
from .kronecker import KroneckerCoefficient
from .partition import Partition
from .permutation import Permutation
from .utils import prod


def all_partitions_of_max_length(n: int, l: Sequence[int], kro: KroneckerCoefficient) -> Iterable[tuple[tuple[Partition, ...], int]]:
    """
    All len(l)-uplets of partitions of n of non-zero Kronecker coefficient
    and so that len(p_i) <= l_i
    """
    # Sort by increasing maximal length (faster) and keep order
    permutation, sorted_l = zip(*sorted(enumerate(l), key=lambda l: l[1]))
    p_inverse = Permutation(permutation).inverse

    # All nuplets of partitions without the last length constraint
    head_product = itertools.product(
        *(Partition.all_for_integer(n, li) for li in sorted_l[:-1])
    )

    # Computing the product and yielding only the partitions of the decomposition
    # whose length respects the last constraint.
    for head in head_product:
        product = kro.product(head)
        for p, c in product.items():
            if len(p) <= sorted_l[-1]:
                yield p_inverse(head + (p,)), c


def all_lambda_matrix(delta: Sequence[int], max_length: NDArray, kro: KroneckerCoefficient) -> Iterable[tuple[NDArray, int]]:
    """ All Lambda matrices form given weight vector and maximal length constraints
    
    Yield a matrix and the product of the Kronecker coefficient of each row.
    """
    N, s = max_length.shape
    assert N == len(delta)

    import numpy as np

    row_product = itertools.product(*(
        all_partitions_of_max_length(n, l, kro)
        for n, l in zip(delta, max_length)
    ))

    for rows_and_coeff in row_product:
        lambda_matrix = np.empty((N, s), dtype=object)
        lambda_matrix[:, :], coeffs = zip(*rows_and_coeff)
        yield lambda_matrix, prod(coeffs)

