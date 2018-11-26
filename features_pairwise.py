##############################################################
#N-Max-Similarity

def Euc_Dist(X,Y):
	return sum( [(X[i]-Y[i])**2 for i in range(len(X))] )**0.5

def Max_Sim(node,list,features,n=5):
	index_node = IDs.index(node)
	indices = [IDs.index(node) for node in list]
	d = sorted([Euc_Dist( features(index_node), features(i) ) for i in indices])
	return d[:n]

