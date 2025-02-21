""" DRAFT

Snippet of parallel computation in main.py
"""
def keep_tau(tau: Tau, Ms: list[Matrix]) -> tuple[bool, Tau]:
    if  tau.is_dom_reg :
        return True, tau
    else: 
        Ms_tau = Lie_action_as_matrices_Vtau(tau, Ms)
        Ms_tauR = [mat_C_to_R(M) for M in Ms_tau]
        return dim_gen_stab_of_K(Ms_tauR) == len(tau.d), tau

def keep_tau2(tau: Tau, Ms: list[Matrix]) -> bool:
    if  tau.is_dom_reg :
        return True
    else: 
        Ms_tau = Lie_action_as_matrices_Vtau(tau, Ms)
        Ms_tauR = [mat_C_to_R(M) for M in Ms_tau]
        return dim_gen_stab_of_K(Ms_tauR) == len(tau.d)
    

    # Filter 2: stabilizer condition
    #from joblib import Parallel, delayed
    #with Parallel(n_jobs=2, backend="multiprocessing") as parallel:
    with Task(f'Step 3: stabilizer condition on {len(Candidates_for_tau1)} tau'):
        ### Avec le nouveau dimStab



        #keep_it = Parallel(n_jobs=6, batch_size=5, backend="multiprocessing")(delayed(keep_tau)(tau, Ms) for tau in Candidates_for_tau1)
        #Candidates_for_tau2 = [tau for tau, mask in zip(Candidates_for_tau1, keep_it) if mask]
        #dummy = Parallel(n_jobs=2, batch_size=5, backend="multiprocessing")(delayed(keep_tau)(tau, Ms) for tau in Candidates_for_tau1) 
        #dummy = parallel(delayed(keep_tau)(tau, Ms) for tau in Candidates_for_tau1) 
        #Candidates_for_tau2 = [tau for keep, tau in dummy if keep]
        
        #Candidates_for_tau2 = [tau for keep, tau in parallel.map(keep_tau, Candidates_for_tau1, itertools.repeat(Ms), chunk_size=5)]

        Candidates_for_tau2 = list(parallel.filter(keep_tau2, Candidates_for_tau1, Ms, chunk_size=5))

        #Candidates_for_tau2 = list(filter(keep_tau, Candidates_for_tau1))
        #Candidates_for_tau2 = [tau for tau in Candidates_for_tau1 if keep_tau(tau, Ms)]