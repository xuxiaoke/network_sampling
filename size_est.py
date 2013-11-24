# -*- coding: utf-8 -*-
"""

A set of graph size estimators introduced in:
[1]    M. Kurant, C. T. Butts and A. Markopoulou, "Graph Size Estimation",  Under submission.

Estimators accepting an independence sample (uniform or weighted):
- estimate_N_NODE
- estimate_N_IND1
- estimate_N_IND2
- estimate_N_IND2_A

Estimators accepting a Random Walk sample: 
- estimate_N_NODE_SimpleThinning (state of the art before [1])
- estimate_N_NODE_ShiftedThinning
- estimate_N_IND_ShiftedThinning                   
- estimate_N_NODE_Margin
- estimate_N_IND_Margin (current state of the art, introduced in [1])

All the implementations guarantee O(n) time complexity, where n is the sample length.

Brief description:
NODE  -- a family of estimators that exploit the nodes repeated in the sample
IND   -- a family of estimators that exploit the edges induced on the sampled nodes
SimpleThinning, ShiftedThinning, Margin  -- different techniques to address the correlations in Random Walks

Example:
>>>


"""
__author__ = """Maciej Kurant"""
__all__ = ['estimate_N_NODE',
           'estimate_N_IND1',
           'estimate_N_IND2',
           'estimate_N_IND2_A',
           'estimate_N_NODE_SimpleThinning',
           'estimate_N_NODE_ShiftedThinning',
           'estimate_N_IND_ShiftedThinning',                   
           'estimate_N_NODE_Margin',
           'estimate_N_IND_Margin'
           ]


####################
class my_counter(dict):
    """Auxiliary. Similar to collections.Counter, 
    but defines differently addition and subtraction.
    
    """
    
    def __init__(self, L):
        dict.__init__(self)
        for l in L: self+=l
        
    def __add__(self,l):
        self[l] = self.get(l,0)+1
        return self
        
    def __sub__(self,l):
        if l not in self:
            raise Exception(str(l)+' not in counter')
        self[l] -= 1
        if self[l]==0:
            del self[l]
        return self

        
################################################################################
###################### Independence Sampling (UIS or WIS) ######################
################################################################################
        
####################
def estimate_N_NODE(S, weights=lambda x:1, separate=False):
    """Graph size estimator. NODE version.

    Graph size estimator following Eqn.(6) in [1]. O(|S|) time complexity. 
    
    Keyword arguments:
    S         -- UIS or WIS sample of graph nodes
    weights   -- weights(v) is the sampling weight of node v in sample S
                 uniform (UIS) by default
    separate  -- if True, returns a pair (numerator,denominator) instead of one number. Default: False
    
    Returns:
        -- the size estimate (float) if available (at least one collision)
        -- -1 otherwise
    
    """    
    
    F1 = sum(weights(v) for v in S)
    F2 = sum(1./weights(v) for v in S)
    F = my_counter(S)
    C = sum(F[v]*(F[v]-1)/2. for v in F if F[v]>1)
    
    if separate:
        return F1*F2, C*2
    else:
        return F1*F2/C/2 if C>0 else -1

####################
def estimate_N_IND1(S, neighbors, weights=lambda x:1):
    """Graph size estimator. IND1 version (performs worse than IND2). 

    Graph size estimator following Eqn.(14) in [1]. O(|S|) time complexity. 
    
    Keyword arguments:
    S         -- UIS or WIS sample of graph nodes
    neighbors -- neighbors(v) iterates over the neighbors of node v
    weights   -- weights(v) is the sampling weight of node v
                 uniform (UIS) by default
    
    Returns:
        -- the size estimate (float) if available (at least one collision)
        -- -1 otherwise
    
    """
        
    F = {}  #good speed-up when many repetitions
    for s in S: F[s] = F.get(s,0.)+1./weights(s)    
    
                       #######     degree    #######            
    volS = sum( F[s] * sum(1. for u in neighbors(s))   for s in F)    
    inv_w = sum(F[s] for s in F) 
    up = inv_w**2 - sum(F[u]*F[u] for u in F)  
    down = sum(F[u]* sum(F[n] for n in neighbors(u) if n in F)  for u in F)    
    
    return -1 if not down else 1 + volS * up/ inv_w /down

####################
def estimate_N_IND2_A(S, A, weights=lambda x:1, separate=False):
    """Graph size estimator, with arbitrary set A given explicitly. 

    Graph size estimator following Eqn.(17) in [1]. O(|S|) time complexity. 
    
    Keyword arguments:
    S         -- UIS or WIS sample of graph nodes
    A         -- an arbitrary set or multiset of nodes 
    weights   -- weights(v) is the sampling weight of node v in sample S
                 uniform (UIS) by default
    separate  -- if True, returns a pair (numerator,denominator) instead of one number. Default: False                 
                 
    Returns:
        -- the size estimate (float) if available (at least one collision)
        -- -1 otherwise
    
    """    
                 
    inv_w = sum(1./weights(s) for s in S)
    
    C = my_counter(A)
    down = sum(1./weights(s)*C.get(s,0) for s in S) 
    
    if separate:
        return inv_w * len(A), down
    else:
        return -1 if not down else inv_w * len(A) / down    
        

####################
def estimate_N_IND2(S, neighbors=None, weights=lambda x:1, unique=True, separate=False):
    """Graph size estimator. IND2 version (better than IND1).

    Graph size estimator following Eqn.(17) in [1], with A defined by Eqn.(18). O(|S|) time complexity. 
    
    Keyword arguments:
    S         -- UIS or WIS sample of graph nodes
    neighbors -- neighbors(v) iterates over the neighbors of node v
    weights   -- weights(v) is the sampling weight of node v in sample S
                 uniform (UIS) by default
    unique    -- if True (default), uses the "set" version of Eqn.(18); typically performs better
    separate  -- if True, returns a pair (numerator,denominator) instead of one number. Default: False
    
    Returns:
        -- the size estimate (float) if available (at least one collision)
        -- -1 otherwise
    
    """    
                 
    inv_w = sum(1./weights(s) for s in S)
    
    if unique:  # "set" version
        A = set(v for s in S for v in neighbors(s))
    else:       # "multiset" version
        A = [v for s in S for v in neighbors(s)]
        
    C = my_counter(A)
    down = sum(1./weights(s)*C.get(s,0) for s in S) 
    
    if separate:
        return inv_w * len(A), down
    else:
        return -1 if not down else inv_w * len(A) / down
        
#####################
#def estimate_N_IND(*args, **kwargs):
#    """The same as estimate_N_IND2. """    
#    
#    return estimate_N_IND2(*args, **kwargs)




################################################################################
###########################  Random Walks  (RW) ################################
################################################################################

####################
def estimate_N_NODE_SimpleThinning(S, weights, th=10):
    """Graph size estimator, NODE version. Accepts a RW sample and decorrelates it with SimpleThinning.

    Graph size estimator following Eqn.(6) in [1] with Eqn.(19). O(|S|) time complexity. 
    Before [1], it was the state of the art in RW-based graph size estimation. 
    
    Keyword arguments:
    S         -- RW sample of graph nodes
    weights   -- weights(v) is the sampling weight of node v in S. weights(v)==deg(v) for RW.
    th        -- thinning parameter                 
    
    Returns:
        -- the size estimate (float) if available (at least one collision)
        -- -1 otherwise
    
    """       
    
    return estimate_N_NODE(S[::th], weights=weights)
        

####################
def estimate_N_NODE_ShiftedThinning(S, weights, th=10):
    """Graph size estimator, NODE version. Accepts a RW sample and decorrelates it with ShiftedThinning.

    Graph size estimator following Eqn.(6) in [1] with Eqn.(22). O(|S|) time complexity. 
    
    Keyword arguments:
    S         -- RW sample of graph nodes
    weights   -- weights(v) is the sampling weight of node v in S. weights(v)==deg(v) for RW.
    th        -- thinning parameter                 
    
    Returns:
        -- the size estimate (float) if available (at least one collision)
        -- -1 otherwise
    
    """       
    
    UP=DOWN=0.
      
    for shift in xrange(th):    
        up,down = estimate_N_NODE(S[shift::th], weights=weights, separate=True)
        UP += up
        DOWN += down
    
    return -1 if not DOWN else UP/DOWN

    
####################
def estimate_N_IND_ShiftedThinning(S, neighbors, weights, th=10):
    """Graph size estimator, IND version. Accepts a RW sample and decorrelates it with ShiftedThinning.

    Graph size estimator following Eqn.(17) in [1] with Eqn.(22). O(|S|) time complexity. 
    
    Keyword arguments:
    S         -- RW sample of graph nodes
    neighbors -- neighbors(v) iterates over the neighbors of node v    
    weights   -- weights(v) is the sampling weight of node v in S. weights(v)==deg(v) for RW.
    th        -- thinning parameter                 
    
    Returns:
        -- the size estimate (float) if available (at least one collision)
        -- -1 otherwise
    
    """         
    
    UP=DOWN=0.
      
    for shift in xrange(th):    
        S_th = S[shift::th]
        A = set()
        for s in S_th:   A.update(neighbors(s))
        up,down = estimate_N_IND2_A(S_th, A, weights=weights, separate=True)
        
        UP += up
        DOWN += down
    
    return -1 if not DOWN else UP/DOWN
    

####################
def estimate_N_NODE_Margin(S, weights, margin=10):
    """Graph size estimator, NODE version. Accepts a RW sample and decorrelates it with Margin-based approach.

    Graph size estimator following Eqn.(23) in [1]. O(|S|) time complexity. 
    
    Keyword arguments:
    S         -- RW sample of graph nodes
    weights   -- weights(v) is the sampling weight of node v in S. weights(v)==deg(v) for RW.
    margin    -- decorrelation margin (pairs of nodes closer than m in the sample are ignored)                
    
    Returns:
        -- the size estimate (float) if available (at least one collision)
        -- -1 otherwise
    
    """  
    n = len(S)
    up = 0.
    down = 0
    C = my_counter(S[margin:])
    W = 1.*sum(weights(s) for s in S[margin:])
    
    for vi, v in enumerate(S):
        if vi-(margin+1)>=0: 
            W += weights(S[vi-(margin+1)])
            C+=S[vi-(margin+1)]
        if vi+margin<n:      
            W -= weights(S[vi+margin])
            C-=S[vi+margin]    
        up += W/weights(v) 
        down += C.get(v,0)         
    
    return -1 if not down else up / down
    
####################
def estimate_N_IND_Margin(S, neighbors, weights, margin=10):
    """Graph size estimator, IND version. Accepts a RW sample and decorrelates it with Margin-based approach.

    Graph size estimator following Eqn.(24) in [1], with additional constraint that A is unique. O(|S|) time complexity. 
    Currently, the state of the art in RW-based graph size estimation. 
    
    Keyword arguments:
    S         -- RW sample of graph nodes
    neighbors -- neighbors(v) iterates over the neighbors of node v    
    weights   -- weights(v) is the sampling weight of node v in S. weights(v)==deg(v) for RW.
    margin    -- decorrelation margin (pairs of nodes closer than m in the sample are ignored)
    
    Returns:
        -- the size estimate (float) if available (at least one collision)
        -- -1 otherwise
    
    """  

    up=0.
    down=0.
    n = len(S)

    A = my_counter(a for s in S[margin:] for a in neighbors(s))
    for si,s in enumerate(S): 
        if si+margin<n: 
            for a in neighbors(S[si+margin]): 
                A[a]-=1
                if not A[a]: del A[a]
        if si-margin>=0: 
            for a in neighbors(S[si-margin]):
                A[a] = A.get(a,0)+1          
                
        ws = weights(s)
        up  += 1./ws * len(A)
        if s in A: 
            down+= 1./ws
        
    return -1 if not down else up / down
    
