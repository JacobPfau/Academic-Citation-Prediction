##############################################################
#N-Max-Similarity

def Euc_Dist(X,Y):
	return sum( [(X[i]-Y[i])**2 for i in range(len(X))] )**0.5

def Max_Sim(node,list,features,n=5):
	index_node = IDs.index(node)
	indices = [IDs.index(node) for node in list]
	d = sorted([Euc_Dist( features(index_node), features(i) ) for i in indices])
	return d[:n]

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
