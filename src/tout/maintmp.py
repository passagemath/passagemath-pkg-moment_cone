from typing import *
from typing import Iterable,Generic,TypeVar
T=TypeVar('T')
load("group.py")
load("root.py")
load("combi.py")
load("weight.py")
load("rep.py")
load("hyperplane_candidates.py")
load("blocks.py")
load("tau.py")

G = LinGroup([5])
V = Representation(G,'fermion',2)
