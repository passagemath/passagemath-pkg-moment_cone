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
    
    def pad(self, length: int) -> tuple[int, ...]:
        """ Returns a padded version of this partition """
        assert length >= len(self._data)
        return self._data + (0,) * (length - len(self._data))

    @staticmethod
    def all_for_integer(n: int) -> Iterable["Partition"]:
        """
        Generates all partitions of an integer n >= 0.

        Could be optimized but it is clearly enough for the n we will consider.
        """
        assert n > 0
        for head in range(n, 0, -1):
            for tail in Partition.all_for_integer(n - head):
                if len(tail) == 0 or head >= tail[0]:
                    yield Partition((head,) + tail._data)

    @staticmethod
    def all_of_height(height: int, alpha: int) -> Iterable["Partition"]:
        """
        Generates all partitions of given height and with given maximum value
        """
        raise NotImplemented
    