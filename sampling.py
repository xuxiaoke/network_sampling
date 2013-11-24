"""
A set of functions to sample nodes of a graph *with* replacements. Also included is a set of corresponding estimators .

Supported sampling methods:
- uniform_independent_node_sample
- random_walk
- metropolis_hastings_random_walk
- weighted_independent_node_sample
- weighted_random_walk

Supported estimators:
- estimate_size
- estimate_mean

Requires free NetworkX library (http://networkx.lanl.gov)

Example:
>>> import networkx as nx
>>> G = nx.generators.wheel_graph(10)
>>> nx.random_walk(G, size=20)
[6, 7, 6, 7, 0, 3, 2, 3, 2, 0, 5, 0, 7, 6, 7, 6, 5, 6, 7, 6]
>>> nx.estimate_size(G, nx.random_walk(G, size=1000), 'random_walk', [0,1,2,3,4])
0.50679999999999814


For more examples try 'test_sampling()'

"""


#Author: Maciej Kurant
#
#Copyright (C) 2004-2010, NetworkX Developers
#Aric Hagberg <hagberg@lanl.gov>
#Dan Schult <dschult@colgate.edu>
#Pieter Swart <swart@lanl.gov>
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are
#met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
#  * Neither the name of the NetworkX Developers nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import networkx as nx
import numpy as np
import random


__author__ = """Maciej Kurant"""
__all__ = ['uniform_independent_node_sample',
           'random_walk', 
           'metropolis_hastings_random_walk',
           'random_walk_stationary_distribution', 
           'weighted_independent_sample',
           'weighted_independent_node_sample', 
           'weighted_random_walk',
           'estimate_size',
           'estimate_mean',  
           'test_sampling']


###############################################################
##################   AUXILIARY FUNCTIONS  #####################
###############################################################

#####################
def __size_type_check(G, size, size_type):
    '''
    Should be treated as private
    '''

    if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
        raise nx.NetworkXException("No support for multigraphs yet!")

    size = int(size)   #just in case, gets crazy when floats are given
    
    if size_type != 'unique' and size_type != 'total':
        raise nx.NetworkXException("size_type must be either 'unique' or 'total'!")
    if size_type=='unique' and size > G.number_of_nodes()/2:
        raise nx.NetworkXException("Too many nodes to collect (more than half in the graph)!")
    return size


############################
def weighted_independent_sample(item_weight, size=1):
    '''
    weighted_independent_sample(item_weight, size=1)

    Returns an independent sample of items drawn with replacements proportionally to their corresponding weights    
    
    Parameters
    ----------
    item_weight:  tuples (item,weight) given in a form of a dictionary, list, or Nx2 np.array
    size:         number of samples to draw 

    Examples
    --------
    >>> qq.weighted_independent_sample({'b':10, 's':2})
    's'
    >>> qq.weighted_independent_sample({'b':10, 's':2}, 5)
    array(['b', 'b', 'b', 'b', 'b'],  dtype='|S1')
    >>> qq.weighted_independent_sample([('s',3),('b',7), ('m',5)], 5)
    array(['b', 'b', 'b', 'b', 'b'],  dtype='|S1')
    >>> qq.weighted_independent_sample( np.array([(1,3),(2,7), (3,5)]), 10)
    array([1, 3, 1, 2, 1, 3, 3, 2, 1, 1])
    
    '''
    
    
    if type(item_weight)==type({}):
        items,weights = zip(*item_weight.items())
        items = np.array(items)
    elif type(item_weight) == list:    # list of couples (item,weight)
        items,weights = zip(*item_weight)
        items = np.array(items)
    elif type(item_weight)==type(np.array([])):
        if len(item_weight.shape)!=2:
            print item_weight
            raise ValueError('item_weight matrix is not 2 dimentional')
        if item_weight.shape[1]==2:
            items = item_weight[:,0]
            weights = item_weight[:,1]
        elif item_weight.shape[0]==2:
            items = item_weight[0,:]
            weights = item_weight[1,:]
        else:
            raise ValueError('item_weight is not 2xn or nx2 matrix')
    else:
        raise ValueError('item_weight not understood')
        
    
    weights_cum = np.cumsum(weights)   
    if size==1:
        R = np.random.rand()*weights_cum[-1]
        return items[np.searchsorted(weights_cum, R)]
    else:
        R = np.random.rand(size)*weights_cum[-1]
        return items[np.searchsorted(weights_cum, R)]


###############################################################
#######################   UNWEIGHTED  #########################
###############################################################

####################
def uniform_independent_node_sample(G, size=10, size_type='total'):
    """
    uniform_independent_node_sample(G, size=10, size_type='total')
    
    Uniform independent sample of nodes
    
    Parameters
    ----------  
    G:            - networkx graph 
    size:         - the sample length (int)
    size_type:  
        'total'   - with repetitions    
        'unique'  - without repetitions    
    """
    
    size = __size_type_check(G, size, size_type)
    
    if size_type == 'total':
        R = np.random.randint(0, G.number_of_nodes(), size)
        return np.array(G.nodes())[R]
    else:
        return random.sample(G.nodes(), size)


####################
def random_walk(G, start_node=None, size=10, size_type='total'):    
    """
    random_walk(G, start_node=None, size=10, size_type='total'):
    
    Returns a list of nodes sampled by the classic Random Walk
    
    Parameters
    ----------  
    G:            - networkx graph 
    start_node:   - starting node (if None, then chosen from the stationary distribution, i.e., proportionally to node degree)
    size:         - the target sample length (int)
    size_type:  
        'total'   - sample length is counted with repetitions    
        'unique'  - sample length is counted without repetitions
    """
    
    size = __size_type_check(G, size, size_type)
    
    if start_node==None:
        start_node = weighted_independent_sample(G.degree())
    
    if size_type=='unique':
        sample_unique = set([start_node])
    
    sample = [start_node]
    while True:
        neighbors = nx.neighbors(G, sample[-1])    
        u = random.choice(neighbors)
        sample.append(u)

        # collected enough samples?
        if size_type=='total':
            if len(sample) >= size:
                break
        else:  #i.e., size_type=='unique'
            sample_unique.add(u)
            if len(sample_unique) >= size:  
                break
    return sample


####################  
def metropolis_hastings_random_walk(G, start_node=None, size=10, size_type='total'):  
    """metropolis_hastings_random_walk(G, start_node=None, size=10, size_type='total')
    
    Returns a list of nodes sampled by the classic Metropolis Hastings Random Walk (with the uniform target node distribution) 
    
    Parameters
    ----------  
    G:            - networkx graph 
    start_node:   - starting node (if none, then chosen uniformly at random)
    size:         - the target sample length (int)
    size_type:  
        'total'   - sample length is counted with repetitions    
        'unique'  - sample length is counted without repetitions
    """

    size = __size_type_check(G, size, size_type)
    
    if start_node==None:
        start_node = random.choice(G.nodes())
        
    sample = [start_node]

    if size_type=='unique':
        sample_unique = set([start_node])
        
    while True:
        u = sample[-1]
        neighbors = nx.neighbors(G, u)

        while True:
            w = random.choice(neighbors)
            if random.random() < float(G.degree(u))/G.degree(w):
                sample.append(w)   # move to w accapted
                break
            else:
                sample.append(u)   # move to w rejected - resample the current node (as if followed a self-loop)
                
        # collected enough samples?
        if size_type=='total':
            if len(sample) >= size:
                sample = sample[:size]  # in case we sampled too many in self-loops
                break
        else: #i.e., size_type=='unique'
            sample_unique.add(w)
            if len(sample_unique) >= size: 
                break
    return sample
    


####################
def random_walk_stationary_distribution(G):
    """random_walk_stationary_distribution(G):
        
    Retuns a dictionary from nodes of G to their stationary distribution probabilities.
    """

    if type(G) != nx.Graph:
        raise nx.NetworkXException("G must be a simple undirected graph!")

    if not nx.is_connected(G):
        raise nx.NetworkXException("G is not connected!")

    tot = 2. * G.number_of_edges()
    pi = {}
    for v in G:
        pi[v] = G.degree(v) / tot
    return pi


###############################################################
#####################  WEIGHTED GRAPH  ########################
###############################################################


####################
def __set_node_weights(G):        #may be useful at some point
    try:            
        for u in G:
            G.node[u]['weight'] = sum(G[u][v]['weight'] for v in G[u]) 
    except:
        raise ValueError("G[u][v]['weight'] probably not defined for some edge")
    return
        
####################
def __weighted_graph_changed(G):

    __set_node_weights(G)

    try:
        G.__WINS_nodes = np.array(G.nodes())
        G.__WINS_weights_cum = np.cumsum( np.array([G.node[v]['weight'] for v in G.__WINS_nodes]) )
    except:
        raise ValueError("G.node[v]['weight'] probably not defined")
            
    G.__node = {}
    for v in G:
        G.__node[v] = {}

####################
def weighted_independent_node_sample(G, size=10, size_type='total', graph_changed=True):
    """
    weighted_independent_node_sample(G, size=10, size_type='total', graph_changed=True)
    
    Weighted independent node sample, with repetitions.
    
    
    Parameters
    ----------  
    G:            - networkx graph (simple). Must have defined weights G[u][v]['weight'] for every edge (u,v).
                    For every node v sets weight G.node[v]['weight'] as a sum of neighboring edge weights.     
    size:         - the target sample length (int)
    size_type:  
        'total'   - sample length is counted with repetitions    
        'unique'  - sample length is counted without repetitions
    graph_changed:
        True      - some auxiliary data structures are created and attached to G, 
                    namely G.__WINS_nodes, G.__WINS_weights_cum, and G.__node.
        False     - assumes that topology and weights of G have not changed since the last call. 
                    reuses G.__WINS_nodes, G.__WINS_weights_cum, and G.__node created above.
                    (this may very significantly speed up the process, especially for large graphs)
    """        
    
    size = __size_type_check(G, size, size_type)
    
    if graph_changed:
        __weighted_graph_changed(G)
    
    if size_type=='total':
        R = np.random.rand(size)*G.__WINS_weights_cum[-1]
        sample = G.__WINS_nodes[np.searchsorted(G.__WINS_weights_cum, R)]
        return sample
    else:          # size_type=='unique'
        R = np.random.rand(size)*G.__WINS_weights_cum[-1]
        sample = list(G.__WINS_nodes[np.searchsorted(G.__WINS_weights_cum, R)])
        while len(set(sample))<size:
            R = np.random.rand(size/2)*G.__WINS_weights_cum[-1]
            sample.extend(G.__WINS_nodes[np.searchsorted(G.__WINS_weights_cum, R)])
        
        while len(set(sample))>size:
            sample.pop()
        return np.array(sample)

####################
def weighted_random_walk(G, start_node=None, size = 10, size_type='total', graph_changed=True):
    """
    weighted_random_walk(G, start_node=None, size = 10, size_type='total', graph_changed=True)
    
    Returns a list of nodes sampled by the Weighted Random Walk
    
    Parameters
    ----------  
    G:            - networkx graph (simple). Must have defined weights G[u][v]['weight'] for every edge (u,v).
                    For every node v sets weight G.node[v]['weight'] as a sum of neighboring edge weights.     
    start_node:   - starting node (if None, then chosen from the stationary distribution, i.e., proportionally to node weight)
    size:         - the target sample length (int)
    size_type:  
        'total'   - sample length is counted with repetitions    
        'unique'  - sample length is counted without repetitions
    graph_changed:
        True      - some auxiliary data structures are created and attached to G, 
                    namely G.__WINS_nodes, G.__WINS_weights_cum, and G.__node.
        False     - assumes that topology and weights of G have not changed since the last call. 
                    reuses G.__WINS_nodes, G.__WINS_weights_cum, and G.__node created above.
                    (this may very significantly speed up the process, especially for large graphs)

    """
    size = __size_type_check(G, size, size_type)
    
    if graph_changed:
        __weighted_graph_changed(G)

    if start_node==None:
        start_node = weighted_independent_node_sample(G, size=1, size_type='total', graph_changed=False)[0]
    
    if size_type=='unique':
        sample_unique = set([start_node])

    sample = [start_node]
    if size==1: 
        return np.array(sample)
                            
    while True:
        u = sample[-1]
        Gu = G.__node[u]
        if not Gu.has_key('W'):
            Gu['W'] = np.cumsum([G[u][v]['weight'] for v in G[u]])
            Gu['N'] = list(G[u])
        i = np.searchsorted(Gu['W'] , random.random()*Gu['W'][-1])
        u = Gu['N'][i]
        sample.append(u)

        if size_type=='total':
            if len(sample)==size:
                break
        else:   # size_type=='unique'
            sample_unique.add(u)
            if len(sample_unique) >= size:
                break

    return np.array(sample)


###############################################################
#######################   ESTIMATORS  #########################
###############################################################

       
        
####################
def estimate_size(G, sample, sample_type, label):
    """
    estimate_size(G, sample, sample_type, label)

    Based on a sample of type 'sample_type', we estimate the relative number of nodes 
    of a given type, i.e., with G.node[v]['label']==label. 
    Parameter 'label' can also be a list/set of nodes of interest. 
    
    Parameters
    ----------  
    G:                 - networkx.Graph 
    sample:            - a node sample
    sample_type:   
        'uniform'      - for samples obtained in UNI or MHRW; a trivial case included for consistency and completeness only
        'random_walk'  - for samples obtained in random walks
        'weighted'     - for samples obtained in weighted_independent_node_sample and weighted_random_walk; uses G.node[v]['weight']
    label:             - type of nodes we consider, i.e., those with with G.node[v]['label']==label
                         if 'label' is a set or list, then it is interpreted as a set of nodes of interest
    
    """
    
    if type(G) != nx.Graph:
        raise nx.NetworkXException("G must be a simple undirected graph!") 

    if sample_type not in ('uniform', 'random_walk', 'weighted'):
        raise nx.NetworkXException("Parameter sample_type '%s' not understood. Use 'uniform', 'random_walk' or 'weighted'." % sample_type) 
        

    values = {}
    if type(label) in (list,set):
        node_category_set = set(label)
        for v in sample:
            if v in  node_category_set:
                values[v] = 1
            else:
                values[v] = 0
    else:    
        for v in sample:
            if G.node[v]['label']==label:
                values[v] = 1
            else:
                values[v] = 0
                
    s = estimate_mean(G, values, sample, sample_type)
    
    if s==0.: 
        return None
    else:
        return s
        
        
    
####################
def estimate_mean(G, values, sample, sample_type, label = None):
    """
    estimate_mean(G, values, sample, sample_type, label = None)
    
    Every node v has some value values[v] attached to it. Based on a sample of type 'sample_type', 
    we estimate and return the average value over all nodes of a given type, i.e., with G.node[v]['label']==label. 
    Parameter 'label' can also be a list/set of nodes of interest.  

    Parameters
    ----------  
    G:                 - networkx.Graph 
    values:            - dictionary: nodes -> values
    sample:            - a node sample
    sample_type:   
        'uniform'      - for samples obtained in UNI or MHRW; a trivial case included for consistency and completeness only
        'random_walk'  - for samples obtained in random walks
        'weighted'     - for samples obtained in weighted_independent_node_sample and weighted_random_walk
    label:             - type of nodes we consider, i.e., those with with G.node[v]['label']==label
                         if 'label' is a set or list, then it is interpreted as a set of nodes of interest
                         label==None indicates all nodes in G (default).
    """

    if type(G) != nx.Graph:
        raise nx.NetworkXException("G must be a simple undirected graph!") 
    
    if sample_type not in ('uniform', 'random_walk', 'weighted'):
        raise nx.NetworkXException("Parameter sample_type '%s' not understood. Use 'uniform', 'random_walk' or 'weighted'." % sample_type) 
        
    
    if label == None:    # overall mean
        sample_in_category = sample
    elif type(label) in (list,set):
        node_category_set = set(label) 
        sample_in_category = [v for v in sample if v in node_category_set]
    else:              # label refers to G.node[v]['label']
        sample_in_category = [v for v in sample if G.node[v]['label']==label]
    
    
    if len(sample_in_category)==0: 
        return None
    else:
        if sample_type=='uniform':
            return sum(1.*values[v] for v in sample_in_category) /  len(sample_in_category)
        elif sample_type=='random_walk':   
            return sum(1.*values[v]/G.degree(v) for v in sample_in_category) / sum(1./G.degree(v) for v in sample_in_category)
        elif sample_type=='weighted':
            return sum(1.*values[v]/G.node[v]['weight'] for v in sample_in_category) / sum(1./G.node[v]['weight'] for v in sample_in_category)
        





###############################################################
###################   EXAMPLE AND TEST  #######################
###############################################################

def __try_size_estimators(G,sample):
    print '%0.3f     %0.3f         %0.3f' % (estimate_size(G, sample, sample_type='uniform', label='in'), estimate_size(G, sample, sample_type='random_walk', label='in'), estimate_size(G, sample, sample_type='weighted', label='in'))


def __try_mean_estimators(G,sample, values, label=None):
    print '%0.3f     %0.3f         %0.3f' % (estimate_mean(G, values, sample, sample_type='uniform', label=label), estimate_mean(G, values, sample, sample_type='random_walk', label=label), estimate_mean(G, values, sample, sample_type='weighted', label=label))

def test_sampling():
        
    G = nx.generators.wheel_graph(10)
    print 
    print nx.info(G)
    print '\nSet random edge weights'
    for u,v in G.edges_iter():
        G[u][v]['weight'] = (u+1)*(v+1)
    
    print 'Set node values to node numbers'
    values={}
    for v in G:
        values[v] = v
    
    print "Label nodes [0,1] and 'in' and others as 'out'." 
    node_category = set([0,1])
    for v in G:
        if v in node_category:
            G.node[v]['label'] = 'in'
        else:
            G.node[v]['label'] = 'out'
    
    
    
    
    N = 10000
    sample_UIS = uniform_independent_node_sample(G, size=N)  # 'uniform' is the correct estimator
    sample_RW  = random_walk(G, size=N)                      # 'random_walk' is the correct estimator
    sample_WIS = weighted_independent_node_sample(G, size=N) # 'weighted' is the correct estimator
    sample_WRW = weighted_random_walk(G, size=N)             # 'weighted' is the correct estimator
    print "\nWe collect %d nodes with the following sample types:" % N    
    print " UIS - uniform_independent_node_sample"
    print " RW  - random_walk"
    print " WIS - weighted_independent_node_sample"
    print " WRW - weighted_random_walk"
    
    print '\nEstimating the relative size of node set [0,1]. The real size is 0.2:'
    print "     uniform   random_walk   weighted    <- 'sample_type' parameter in the estimator"
    print 'UNI ',
    __try_size_estimators(G,sample_UIS)
    print 'RW  ', 
    __try_size_estimators(G,sample_RW)
    print 'WIS ',
    __try_size_estimators(G,sample_WIS)
    print 'WRW ',
    __try_size_estimators(G,sample_WRW)
    
    
    print '\nEstimating mean value. The real mean is 4.5:'
    print "     uniform   random_walk   weighted     <- 'sample_type' parameter in the estimator"
    print 'UNI ',
    __try_mean_estimators(G,sample_UIS, values)
    print 'RW  ', 
    __try_mean_estimators(G,sample_RW, values)
    print 'WIS ',
    __try_mean_estimators(G,sample_WIS, values)
    print 'WRW ',
    __try_mean_estimators(G,sample_WRW, values)
    
    
    print '\nEstimating mean value inside [0,1]. The real mean is 0.5:'
    print "     uniform   random_walk   weighted    <- 'sample_type' parameter in the estimator"
    print 'UNI ',
    __try_mean_estimators(G,sample_UIS, values, 'in')
    print 'RW  ', 
    __try_mean_estimators(G,sample_RW, values, 'in')
    print 'WIS ',
    __try_mean_estimators(G,sample_WIS, values, 'in')
    print 'WRW ',
    __try_mean_estimators(G,sample_WRW, values, 'in')
    
    
    print "\n 'label' can be a real label corresponding to G.node[v]['label'], or a set of nodes of interest:"
    __try_mean_estimators(G,sample_WRW, values, 'in')
    __try_mean_estimators(G,sample_WRW, values, [0,1])
    __try_mean_estimators(G,sample_WRW, values, set([0,1]))


test_sampling()



#####################
#def RW_transition_matrix(G):
#    """
#    return P = D*A, i.e., the transition matrix of Random Walk
#    where:
#     D - a diagonal matrix of 1/degree
#     A - adjacency matrix
#    """
#    n = G.number_of_nodes()
#    D = np.zeros((n,n))
#    for i,v in enumerate(G): D[i][i]=1./G.degree(v)
#    A = nx.adj_matrix(G)
#    return D*A
#    
#####################
#def RW_symmetric_transition_matrix(G):
#    """
#    return P = D^0.5 * A * D^0.5
#    where:
#     D - a diagonal matrix of 1/degree
#     A - adjacency matrix
#    This is a symmetric matrix, as described in L. Lovasz 'Random Walks on Graphs : A Survey', page 15
#    The symmetry simplifies the spectral analysis.
#    """
#    n = G.number_of_nodes()
#    D = np.zeros((n,n))
#    for i,v in enumerate(G): D[i][i]=1./G.degree(v)
#    A = nx.adj_matrix(G)
#    return D**(0.5) * A * D**(0.5)
#
#        
#####################
#def MHRW(G, start_node=None, size=10, size_type='unique'):  
#    '''
#    MHRW(G, start_node=None, size=10, size_type='unique')
#    
#    Shorthand for 
#    metropolis_hastings_random_walk(G, start_node=None, size=10, size_type='unique')
#    
#    '''
#    return metropolis_hastings_random_walk(G, start_node, size, size_type):  
#    
#
#    
#
#
#####################
#def MHRW_compressed(G, v, max_len):   
#    """
#    max_len  -  max number of unique nodes
#    Speeded up selfloops by drawing from geometrical distribution
#    Returns dict of node frequencies (occurencies) rather then the entire walk
#    """
#    
#    if max_len > G.number_of_nodes()/2:
#        raise ValueError ("Too many nodes to collect!")
#
#    v_last = v
#    V = {v:1}
#    while len(V) < max_len:
#        neighbors = nx.neighbors(G, v_last)
#        d_last = G.degree(v_last)
#        
#        if len(neighbors)<10:            
#            p = min(1., MHRW_prob_of_leaving_v(G,v_last))  # sometimes rounding error            
#            k = np.random.geometric(p)
#            A = [(n, min(float(d_last)/G.degree(n), 1.) / (p * d_last)) for n in neighbors]     # list of neighbors and their probabs of being chosen (sums to 1)            
#            v_new = A[np.searchsorted( np.cumsum(zip(*A)[1]), random.random())]
#            
#            V[v_last] = V.get(v_last,0) + k  -1
#            v_last = v_new[0]
#            V[v_last] = V.get(v_last,0) + 1
#            
#        else:
#            while True:
#                w = random.choice(neighbors)
#                if random.random() < float(d_last)/G.degree(w):
#                    v_last = w
#                    V[w] = V.get(w,0) + 1
#                    break
#                else:
#                    V[v_last] = V.get(v_last,0) + 1
#    return V
#    
#####################
#def MHRW_prob_of_leaving_v(G,v):
#    return sum([min(1.,float(G.degree(v))/d)/G.degree(v) for d in G.degree(G.neighbors(v))])
#
#
#####################
#def MHRW_stay_in_v(G,v):
#    ex = 1./MHRW_prob_of_leaving_v(G,v)
#    q = int(ex)# +1
#    if random.random() < ex-int(ex): q+=1
#    return q
