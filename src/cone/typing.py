from sage.all import Integer as SageInteger
from typing import TypeVar, TypeAlias, Generic, Optional, Iterable, Iterator, cast
from collections.abc import Sequence

Integer = TypeVar('Integer', int, SageInteger)
T = TypeVar('T')
