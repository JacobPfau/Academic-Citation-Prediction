import numpy as np

##############################################################
#N-Max-Similarity

def Euc_Dist(X,Y):
	return sum( [(X[i]-Y[i])**2 for i in range(len(X))] )**0.5

def Max_Sim(source_ID,target_ID,features,graph,node_dict,n=5):
    index_source = node_dict[source_ID]
    citers_graph_indices = graph.predecessors(target_ID)
    citers_IDs = [graph.vs[i].attributes()['name'] for i in citers_graph_indices]
    indices_citers = [node_dict[node] for node in citers_IDs]
    d = sorted([Euc_Dist( features[index_source], features[i] ) for i in indices_citers],reverse=True)
    if len(d)>=n:
    	return np.array(d[:n])
    else:
    	d.extend([0 for i in range(n-len(d))])
    	return np.array( d )

##############################################################
#Papers with similar abstracts that also cite the target node

def Get_K_NN(node_ID,kdtree,features,node_dict,index_dict,k_val=20):
    """
    @features param3: the array of features i.e. points which were used to build the kdtree
    @node_dict param4: keys are node_IDs, values are node indices
    @index_dict param5: keys are node indices, values are node_IDs
    @return: pair (indices,distances) of k_val nearest abstracts
    """

    index_node = node_dict[node_ID]
    point = features[index_node]
    dist, ind = kdtree.query([point], k=k_val)
    IDs = [index_dict[i] for i in ind[0]] 
    return (IDs,dist)

def Citation_Check(source_ID,target_ID,kdtree,features,graph,node_dict,index_dict,k=20):
	"""
    @features param4: the array of features i.e. points which were used to build the kdtree
	@return: numpy array [% KNN of source which cite target, % KNN of target which are cited by source]
	"""
	close_source = Get_K_NN(source_ID,kdtree,features,node_dict,index_dict,k_val=k)[0]
	close_target = Get_K_NN(target_ID,kdtree,features,node_dict,index_dict,k_val=k)[0]
	cite_percent = len([n for n in close_source if graph.are_connected(n,target_ID)])/len(close_source)
	cited_percent = len([n for n in close_target if graph.are_connected(source_ID,n)])/len(close_target)

	return np.array([cite_percent,cited_percent])



##############################################################
# Peer popularity
#
# Estimates the "popularity" of node c in the surrounding of node n.
# Returns the percentange of how many (back & forward) neighbours of n cite c?
# Let N be all nodes w such that an edge (w,n) or (n,w) exists.
# Return value: |{w in N s.t. exists edge (w,c)}| / |N|
def peer_popularity(graph, source_ID, target_ID):
    cites = 0
    for w in graph.neighbors(source):
        if graph.are_connected(w,target):
            cites += 1
    return cites/graph.neighborhood_size(source)
