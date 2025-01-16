class Representation:
    def __init__(self, G : LinGroup, type : str, nb_part=None):
        """
        Initializes an instance of Representation.
        
        :param G: An instance of the LinGroup class.
        :param type: A string, must be 'kron', 'fermion', or 'boson'.
        :param nb_part: An integer, required if type is 'fermion' or 'boson'.

        Example : 
        >>> V=Representation(G,'fermion',2)
        >>> V
        Representation(G=GL(5), type='fermion', number of particules=2)
        >>> V.dim
        10
        >>> V.G
        GL(5)
        >>> V.type
        'fermion'
        """
        
        if not isinstance(G, LinGroup):
            raise TypeError("G must be an instance of the Group class.")
        
        if type not in ['kron', 'fermion', 'boson']:
            raise ValueError("type must be 'kron', 'fermion', or 'boson'.")
        
        self.G = G
        self.type = type

        if type in ['fermion', 'boson']:
            if nb_part is None :
                raise ValueError("nb_part must be an integer for types 'fermion' or 'boson'.")
        self.nb_part = nb_part
        
    
    def __repr__(self):
        """
        String representation of the instance.
        """
        if self.type in ['fermion', 'boson']:
            return f"Representation(G={self.G}, type='{self.type}', number of particules={self.nb_part})"
        return f"Representation(G={self.G}, type='{self.type}')"

    @cached_property
    def dim_cone(self) -> int:
        """
        Expected dimension of the cone. To be checked with Stabilizer of K.
        
        >>> V=Representation(G,'fermion',2)
        >>> V.dim_cone
        5
        """
        
        if self.type == 'kron':
            return self.G.rank-len(self.G)+1 
        else :
            return self.G.rank

    @cached_property
    def dim(self) -> int:
        """
        Dimension of V.
        """
        if self.type == 'kron':
            return prod(self.G) 
        elif self.type == 'fermion':
            return binomial(self.G.rank,self.nb_part)
        else :
            return binomial(self.G.rank+self.nb_part-1,self.G.rank-1)
        
    @cached_property    
    def all_weights(self) -> Iterable["Weight"] :
        """
        Create de the list of weights of T acting on V. 

      >V=Representation(G,'fermion',2)
      >L=[chi for chi in V.all_weights]
      >L
      [A list of epsilon_i for each copy of GL: [[0, 1]],
      A list of epsilon_i for each copy of GL: [[0, 2]],
      A list of epsilon_i for each copy of GL: [[0, 3]],
      A list of epsilon_i for each copy of GL: [[0, 4]],
      A list of epsilon_i for each copy of GL: [[1, 2]],
      A list of epsilon_i for each copy of GL: [[1, 3]],
      A list of epsilon_i for each copy of GL: [[1, 4]],
      A list of epsilon_i for each copy of GL: [[2, 3]],
      A list of epsilon_i for each copy of GL: [[2, 4]],
      A list of epsilon_i for each copy of GL: [[3, 4]]]
        """
        
        if self.type == 'kron' :
            for i,w in enumerate(itertools.product(*(range(di) for di in self.G))):
                yield Weight(self.G,as_list=list(w),index=i)
        elif self.type == 'fermion' :
            for i,w in enumerate(itertools.combinations(range(self.G[0]),self.nb_part)):
                print('w',[list(w)])
                yield Weight(self.G,as_list_of_list=[list(w)],index=i)
        else :
            for i,w in enumerate(itertools.combinations(range(self.G[0]+self.nb_part-1),self.G[0]-1)) :
                v=vector(ZZ,self.G.rank)
                v[0]=w[0]
                for i,p in enumerate(itertools.pairwise(w)):
                    v[i+1]=p[1]-p[0]-1
                v[-1]=self.G.rank-w[-1]-1
                yield Weight(self.G,as_vector=v,index=i)


    def weights_of_S(self, p : Partition) -> Iterable["Weight"] : # Could be improved
        """
        Create de the list of weights of S\subset T acting on V. With multiplicities.
        S is given by p.
        Only used for V.type='fermion' or 'boson'
        """
        if self.type not in ['fermion','boson']:
            raise TypeError("Weights_of_S is only implemented for fermions")

        GS=LinGroup([len(p)])
        List_indices = []
        for i,x in enumerate(p) :
            List_indices += x*[i]
        
        chiTot=Weight(GS,as_list_of_list=[List_indices])
        for i,l in enumerate(unique_combinations(List_indices, self.nb_part)):
            chi = Weight(GS,as_list_of_list=[l],index=i)
            chi.mult=prod([binomial(chiTot.as_vector[i],chi.as_vector[i]) for i in range(len(p))])
            yield chi


    #@staticmethod         TODO : il faut que cette matrice soit calculée une seule fois pour chaque alpha
    def rhoEij(self,alpha : Root) -> matrix:
        """
        Return the matrix rho_V(E_alpha)
        """
        M = matrix(QQ[I],self.dim,self.dim)
        if self.type == 'kron' : 
            Gred=LinGroup(self.G[:alpha.k]+self.G[alpha.k+1:])
            Vred=Representation(Gred,'kron')
            for w in Vred.all_weights:
                wj = Weight(self.G,as_list=list(w.epsi[:k])+[alpha.j]+list(w.epsi[k:]))
                idj=wj.index(selft) 
            if alpha.i==alpha.j:
                M[idj,idj]=1
            else :    
                wi = Weight(self.G,as_list=list(w.epsi[:k])+[alpha.i]+list(w.epsi[k:]))
                idi=wi.index(self)
                M[idi,idj]=1
        elif self.type == 'fermion': # TODO : rassembler fermion et bosons car bcp de commun
            Vred=Representation(LinGroup([self.G[0]-1]),'fermion',self.nb_part-1)
            for w in Vred.all_weights:
                L1=[s for s in w.as_list_of_list[0] if s<alpha.j]
                L2=[s+1 for s in w.as_list_of_list[0] if s>=alpha.j]
                lj=L1+[alpha.j]+L2  # we insert j
                if alpha.i not in lj : # Otherwise E_ij v =0
                    wj = Weight(self.G,as_list_of_list=[lj])
                    idj=wj.index(self)
                    #print('idj',idj)
                    if alpha.i==alpha.j:
                        M[idj,idj]=1
                    else :    
                        li=L1+[alpha.i]+L2  # we insert i
                        li.sort()
                        wi = Weight(self.G,as_list_of_list=[li])
                        idi=wi.index(self)
                        M[idi,idj]=1
        else:
            Vred=Representation(self.G,'boson',self.nb_part-1)
            for w in Vred.all_weights:
                v=w.as_vector
                v[alpha.j]+=1
                wj = Weight(self.G,as_vector=v)
                idj=wj.index(self)
                if alpha.i==alpha.j:
                    M[idj,idj]=1
                else :
                    v[alpha.j]-=1
                    v[alpha.i]+=1
                    wi = Weight(self.G,as_vector=v)
                    idi=wi.index(self)
                    M[idi,idj]=1

        return M

    @cached_property
    def actionK(self) -> list[matrix]:
        """
        The list of matrices rho_V(xi) for xi in the bases of K.
        """
        L=[]
        for beta in Root.all_of_K(self.G) :
            k,i,j = beta.k,beta.i,beta.j
            if i == j :
                L.append(I*self.rhoEij(beta))
            elif i<j:
                L.append(self.rhoEij(beta)-self.rhoEij(beta.opposite))
            else :
                L.append(I*self.rhoEij(beta)+I*self.rhoEij(beta.opposite))
        return(L)

    def u_max(self, Vred: "Representation") -> int:
        """
        Maximal value of u obtained by extending a e-1-PS to a d-1-PS

        For a dimension vector d=(d_i), and a list of number of Levi blocks in each d_i,
        computes the maximal dimension of a nilradical.

        Examples:
        >>> d1 = Dimension((2, 3, 4))
        >>> d1.u_max(d1)
        10
        >>> d2 = Dimension((3, 5, 2))
        >>> d2.u_max(d2)
        14
        >>> d1.u_max(d2)
        8
        >>> d2.u_max(d1)
        11
        """
        from math import floor
        # TODO assert V.type='kron'
        return sum(floor(d * d / 2 * (1 - 1 / e)) for d, e in zip(self.G, Vred.G))
        
