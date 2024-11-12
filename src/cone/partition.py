from .typing import *

class Partition(Sequence[Integer]):
    """ Decreasing sequence of positive integer """
    __slots__ = "_data",
    _data: tuple[Integer, ...]

    def __init__(self, p: Sequence[Integer]):
        from .utils import is_decreasing, trim_zeros
        assert is_decreasing(p) and p[-1] >= 0, "Invalid partition"

        # Auto trim the partition
        self._data = tuple(trim_zeros(p))

    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, idx: int) -> Integer:
        return self._data[idx] if idx < len(self) else cast(Integer, 0)
    
    def __iter__(self) -> Iterator[Integer]:
        return iter(self._data)
    
    def __repr__(self) -> str:
        return f"Partition({self._data})"
    
    