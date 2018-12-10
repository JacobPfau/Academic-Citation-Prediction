import numpy as np
from sklearn.metrics.pairwise import cosine_distances as COS
from functools import reduce
import math

##############################################################
#N-Max-Similarity (by L2 on LSA from source to papers which cite target)

def Euc_Dist(X,Y):
	return sum( [(X[i]-Y[i])**2 for i in range(len(X))] )**0.5

def Max_Sim(source_ID,target_ID,features,graph,node_dict,metric="COS",n=3):
	index_source = node_dict[source_ID]
	citers_graph_indices = graph.predecessors(target_ID)
	citers_IDs = [graph.vs[i].attributes()['name'] for i in citers_graph_indices]
	indices_citers = [node_dict[node] for node in citers_IDs]
	if metric=="L2":
		d = sorted([Euc_Dist( features[index_source], features[i] ) for i in indices_citers],reverse=True)
	if metric=="COS":
		d = sorted([COS( features[index_source].reshape(1,-1), features[i].reshape(1,-1) )[0][0] for i in indices_citers],reverse=True)
	if len(d)<=2*n:
		d.extend([0 for i in range(2*n-len(d))])
	d1=d[:n]
	d1.extend(d[-n:])
	d1.extend([np.mean(d[:n]),np.mean(d[-n:]),np.mean(d)])
	return np.array(d1)

##############################################################
#Papers with similar abstracts that also cite the target node

def Get_K_NN(node_ID,kdtree,features,node_dict,index_dict,k_val=1000):
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

def Citation_Check(source_ID,target_ID,kdtree,features,graph,node_dict,index_dict,k=500):
    """
    @features param4: the array of features i.e. points which were used to build the kdtree
    @return: numpy array [% KNN of source which cite target, % KNN of target which are cited by source]
    """
    close_source = Get_K_NN(source_ID,kdtree,features,node_dict,index_dict,k_val=k)[0]
    close_target = Get_K_NN(target_ID,kdtree,features,node_dict,index_dict,k_val=k)[0]
    cite_percent = len([n for n in close_source if graph.are_connected(n,target_ID)])/len(close_source)
    w_s_p = sum([math.log2(500-i+1) for i,x in enumerate(close_source) if graph.are_connected(x,target_ID)])/5500
    w_t_p = sum([math.log2(500-i+1) for i,x in enumerate(close_target) if graph.are_connected(source_ID,x)])/5500
    cited_percent = len([n for n in close_target if graph.are_connected(source_ID,n)])/len(close_target)

    return np.array([cite_percent,cited_percent,w_s_p,w_t_p])


##############################################################
# Peer popularity
#
# Estimates the "popularity" of node c in the surrounding of node n.
# Returns the percentange of how many (back & forward) neighbours of n cite c?
# Let N be all nodes w such that an edge (w,n) or (n,w) exists.
# Return value: |{w in N s.t. exists edge (w,c)}| / |N|
def peer_popularity(graph, source_ID, target_ID):
    cites = 0
    for w in graph.neighbors(source_ID):
        if graph.are_connected(w,target_ID):
            cites += 1
    return cites/graph.neighborhood_size(source_ID)

##############################################################
# Share edge

def edge_check(source_ID, target_ID, graph):
	return int(graph.get_eid(source_ID,target_ID,directed=True,error=False)!=-1)

##############################################################
# LSA similarity

def LSA_distance(source_ID,target_ID,node_dict,LSA_array,metric="COS"):
	index_source = node_dict[source_ID]
	index_target = node_dict[target_ID]
	if metric=='COS':
		return COS(LSA_array[index_source].reshape(1,-1),LSA_array[index_target].reshape(1,-1))[0][0]
	if metric=='L2':
		return Euc_Dist(LSA_array[index_source],LSA_array[index_target])

##############################################################
#Node degree
"""
@returns: length 4 array containing IN degree and OUT degree for source and target
"""

def node_degree(source_ID,target_ID,graph):
	return np.array([graph.degree(source_ID,mode='IN'),graph.degree(source_ID,mode='OUT'),graph.degree(target_ID,mode='IN'),graph.degree(target_ID,mode='OUT')])

##############################################################
#Successors(source) intersect successors(predecessors of target) etc.
"""
"""

def succ_pred(source_ID,target_ID,graph):
	succ_source = set(graph.successors(source_ID))

	pred_target = graph.predecessors(target_ID)
	pred_IDs = [graph.vs[i].attributes()['name'] for i in pred_target if graph.vs[i].attributes()['name']!=source_ID]
	pred_succ = [set(graph.successors(id)) for id in pred_IDs]

	inter = [succ_source & p_s for p_s in pred_succ]
	union = [succ_source | p_s for p_s in pred_succ]
	len_union = [len(x) for x in union]
	len_inter = [len(x) for x in inter]

	jacc = [len_inter[i]/len_union[i] for i in range(len(len_union))]
	if len(inter)>0:
		total_inter = len(reduce(lambda x,y:x|y,inter))
	else:
		total_inter = 0
	if len(len_inter)>0:
		stats = [max(len_inter),np.mean(len_inter),total_inter,max(jacc)]
	else:
		stats = [0,0,total_inter,0]
	return np.array(stats)

##############################################################
#Successors(source) intersect successors(predecessors of target) etc.
"""
@return: np.array([title intersection,time difference, author intersection])
"""

def baseline(source_ID,target_ID,node_dict,node_info):
    f=[]
    
    index_source = node_dict[source_ID]
    index_target = node_dict[target_ID]
    source_info = node_info[index_source]
    target_info = node_info[index_target]
    # convert to lowercase and tokenize
    source_title = source_info[2].lower().split(" ")
    # remove stopwords
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]

    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",") 

    overlap_title = len(set(source_title).intersection(set(target_title)))
    temp_diff = int(source_info[1]) - int(target_info[1])
    comm_auth = len(set(source_auth).intersection(set(target_auth)))
    
    f.append(len(set(source_title).intersection(set(target_title))))
    f.append(int(source_info[1]) - int(target_info[1]))
    f.append(len(set(source_auth).intersection(set(target_auth))))
    
    return np.array(f)