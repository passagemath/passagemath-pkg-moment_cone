import itertools
from functools import cached_property

from .root import Root
from .weight import Weight
from .inequality import Inequality


class Monomial:
    """
    A monomial is a list of roots that represent de product of the corresponding variables in the ring QV.
    We allow toral roots [k,i,i] that correspondant to 1's in the monomial.
    A monomial has a degree.
    """
    roots: list[Root]

    def __init__(self, roots: list[Root]):
        """
        Initialize a Monomial object.

        :param roots: List of Root objects.
        """
        self.roots = roots

    @cached_property
    def degree(self) -> int:
        """ Calculate the default degree, excluding toral roots [k, i, i] """
        return sum(1 for root in self.roots if root.i != root.j)

    def __repr__(self) -> str:
        """
        String representation of the Monomial.
        """
        roots_repr = ", ".join(str(root) for root in self.roots)
        return f"Monomial(roots=[{roots_repr}], degree={self.degree})"


def Is_Max_un(l: list[int]) -> bool:
    """
    Check if the maximum of a list of integers is at most 1 or not.
    """
    return all(x <= 1 for x in l)


def List_var_1(l : list[Monomial]) -> list[Root]:
    """
    Given a list of monomials, return the list of roots that give linear equalities. 
    """
    ll=[]
    for m in l :
        if m.degree==1 : 
           for beta in m.roots:
               if beta.i!=beta.j and beta not in ll :
                   ll.append(beta)
                   
    return(ll)


def Is_Lin_Triangular(equa: dict[int, list[Monomial]]) -> bool: #equa is dictionnary (index of a Weight) -> list of Monomials
    """
    Check if a given list of equalities is Linear Trinagular or not.
    Each equality is actually given by its list of monomials (without coefficients)
    """
    if len(equa.keys())==0 : return(True)
    List_eq_lin=[]
    list_var_deg_1=[]
    for id_p in equa.keys():
        if Is_Max_un([m.degree for m in equa[id_p]]) :
            List_eq_lin.append(id_p)
            for beta in List_var_1(equa[id_p]):
                if beta not in list_var_deg_1 :
                    list_var_deg_1.append(beta)
    
    if len(List_eq_lin)==0 :
        return False
    
    if len(List_eq_lin)>len(list_var_deg_1):
        raise ValueError('Trop ineg lin')   
    
    if len(List_eq_lin)!=len(list_var_deg_1) :
        return False
       
    # We put the linear variables at 0
    newequa={}
    for id_p in equa.keys():
        if id_p not in List_eq_lin :
            lmon=[]
            for x in equa[id_p]:
                if not(set(x.roots) & set(list_var_deg_1)) :
                    #nx=x.copy()
                    lmon.append(x)               
            newequa[id_p]=lmon
    return Is_Lin_Triangular(newequa)


def Fiber_LT(ineq: Inequality) -> bool:
    """
    Input : an inequality. 
    Return True iff the equations of the fibers of pi are linear triangular.
    """
    tau=ineq.tau
    d=ineq.tau.d
    # Inversions of U(w)xT with a dictionnary pos --> list of roots
    inv_w_dic={}
    for k in range(len(d)):
        inv_w_dic[k]=[Root(k,i,i) for i in range(d[k])]                
    for beta in ineq.inversions:
        inv_w_dic[beta.k].append(beta)    

    # Weights with positive scalar with ineq.tau    
    Poids_pos=[]
    for ll in list(tau.positive_weights.values()):
        Poids_pos+=ll

    # Weights with non positive scalar with ineq.tau       
    Poids_neg0=[]
    for ll in list(tau.non_positive_weights.values()):
        Poids_neg0+=ll

    #Initialization of equa as the dictionnary positive weight --> empty list    
    equa: dict[int, list[Monomial]] = {}
    for p in Poids_pos:
        equa[p.index_in(d)] = []
    
    for betas in itertools.product(*inv_w_dic.values()):
        weight_new = Weight([beta.i for beta in betas])
        weight_old = Weight([beta.j for beta in betas])
        if (weight_new in Poids_pos) and (weight_old in Poids_neg0) :
            equa[weight_new.index_in(d)].append(Monomial(list(betas)))

    return Is_Lin_Triangular(equa)
       
