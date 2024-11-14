from .typing import *

class Partition(Sequence[int]):
    """ Decreasing sequence of positive int """
    __slots__ = "_data",
    _data: tuple[int, ...]

    def __init__(self, p: Sequence[int]):
        from .utils import is_decreasing, trim_zeros
        assert is_decreasing(p) and p[-1] >= 0, "Invalid partition"

        # Auto trim the partition
        self._data = tuple(trim_zeros(p))

    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, idx: int) -> int:
        return self._data[idx] if idx < len(self) else cast(int, 0)
    
    def __iter__(self) -> Iterator[int]:
        return iter(self._data)
    
    def __repr__(self) -> str:
        return f"Partition({self._data})"
    
    