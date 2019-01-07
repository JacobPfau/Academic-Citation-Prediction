import numpy as np
from sklearn.metrics.pairwise import cosine_distances as COS
from functools import reduce
import math
import copy
import multiprocessing as mp
import multi_func as mf
import preprocessing as prep


def by_chunk(pairs_array, gold_graph, kdtree, features, node_dict, index_dict, pairs_subset_edges=True, chunk_size=1000, to_do = {'succ_pred':True, 'max_sim':True, 'citation_check':True, 'node_degree':True, 'reverse_max_sim':True}, k_cc=500, metric_ms='COS', n_ms=3):
    num_chunks = math.ceil(len(pairs_array)/chunk_size)
    l = len(pairs_array)
    sp = []
    ms = []
    cc = []
    nd = []
    rm = []

    for k in range(num_chunks):
        chunk = pairs_array[ k*chunk_size : min([(k+1)*chunk_size,len(pairs_array)]) ]
        chunk_removed_graph = copy.copy(gold_graph)

        if pairs_subset_edges==True:
            to_del = []
            for triple in chunk:
                source = triple[0]
                target = triple[1]
                if triple[2]=='1':
                    to_del.append((source,target))
            chunk_removed_graph.delete_edges(to_del)

        for triple in chunk:
            source_ID = triple[0]
            target_ID = triple[1]
            if to_do['succ_pred']==True:
                sp.append(succ_pred(source_ID,target_ID,chunk_removed_graph))
            if to_do['max_sim']==True:
                ms.append(Max_Sim(source_ID,target_ID,features,chunk_removed_graph,node_dict,metric=metric_ms,n=n_ms))
            if to_do['citation_check']==True:
                cc.append(Citation_Check(source_ID,target_ID,kdtree,features,chunk_removed_graph,node_dict,index_dict,k=k_cc))
            if to_do['node_degree']==True:
                nd.append(node_degree(source_ID,target_ID,chunk_removed_graph))
            if to_do['reverse_max_sim']==True:
                rm.append(Reverse_Max_Sim(source_ID, target_ID, features, chunk_removed_graph, node_dict))

        print(k, '/', num_chunks)

    return (sp,ms,cc,nd,rm)




def path_length(source_ID, target_ID, paths_dict):
    # return the path length from source to target or 35 if no path exists (i.e. a length longer than any observed min. path)
    return min([paths_dict[source_ID][target_ID],35])

##############################################################
#temp_match
#How well does the publication year of source fit to the years of other papers that cited target?
def temp_fit(source_ID, target_ID, graph, node_dict, publication_years):
    source_year = int(publication_years[node_dict[source_ID]])
    pred_target = graph.predecessors(target_ID)
    pred_IDs = [graph.vs[i].attributes()['name'] for i in pred_target if graph.vs[i].attributes()['name']!=source_ID]
    pred_years = np.array([int(publication_years[node_dict[pred_id]]) for pred_id in pred_IDs])
    
    if len(pred_IDs) == 0:
        # there are no citations. This should occur rarely
        # Set values worse than anything observed
        return [20,0]
    
    #distance to closest two years with citations
    if len(pred_IDs) == 1:
        closest2 = min(abs(pred_years-source_year))
    else:
        closest2 = np.mean(np.partition(abs(pred_years-source_year),1)[0:2])
    #How many citations within +- 1 year of source_year
    one_year_corridor = len([1 for y in pred_years if (source_year-1 <= y and y <= source_year+1)])/len(pred_IDs)    
        
    return [closest2,one_year_corridor]

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
#Reverse-N-Max-Similarity (by L2 on LSA from target to papers which are cited by source)

def Reverse_Max_Sim(source_ID,target_ID,features,graph,node_dict,metric="COS",n=3):
    index_target = node_dict[target_ID]
    citees_graph_indices = graph.successors(source_ID)
    citees_IDs = [graph.vs[i].attributes()['name'] for i in citees_graph_indices]
    indices_citees = [node_dict[node] for node in citees_IDs]
    if metric=="L2":
        d = sorted([Euc_Dist( features[index_target], features[i] ) for i in indices_citees],reverse=True)
    if metric=="COS":
        d = sorted([COS( features[index_target].reshape(1,-1), features[i].reshape(1,-1) )[0][0] for i in indices_citees],reverse=True)
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


################################################
# Successors(source) intersect successors(predecessors of target) etc.

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



###################################################################
# compute_all_features
#
# Wrapper to compute all our features
# Returns a dict containing a numpy array for each feature
#
def compute_all_features(node_pair_set, 
                         graph, 
                         IDs,
                         node_info,
                         stemmer,
                         stpwds,
                         kdtree, 
                         l, 
                         l_ngrams,
                         t_titles,
                         node_dict, 
                         index_dict,
                         pairs_subset_edges,
                         publication_years):
    
    features_to_create = ['overlap_title',
                      'comm_auth',
                      'temp_diff',
                      'citation_check',
                      'max_sim',
                      'peer_popularity',
                      'succ_pred',
                      'LSA_distance',
                      'title_sim',
                      'temporal_fit',
                      'N_LSA_distance',
                      'path_length',
                      'node_degree',
                      'reverse_max_sim']
    
    # We insert features in a dictionary
    insert_features_dict = dict()
    for feat in features_to_create:
        insert_features_dict[feat] = []
    set_to_use = node_pair_set

    # Compute some features in parallelized chunks.
    p = mf.params(graph, kdtree, l, node_dict, index_dict, pairs_subset_edges, chunk_size = 1000)
    grouped_set = [set_to_use[2000*i:2000*(i+1)] for i in range(math.ceil(len(set_to_use)/2000))]
    pool = mp.Pool(mp.cpu_count())
    path_dict_list = pool.map(p.all_paths_noparams, grouped_set, chunksize = 10)
    chunked_output = zip(*pool.map(p.by_chunk_noparams, grouped_set, chunksize = 10))
    pool.close()

    # Recombine chunked output
    all_path_dict = dict()
    for i in IDs:
        all_path_dict[i] = dict()

    for d in path_dict_list:
        for source_id in d:
            for target_id in d[source_id]:
                all_path_dict[source_id][target_id] = d[source_id][target_id]

    list_chunked_output = list(chunked_output)
    for feature_list in list_chunked_output[0]:
        insert_features_dict['succ_pred'].extend(feature_list)
    for feature_list in list_chunked_output[1]:
        insert_features_dict['max_sim'].extend(feature_list)
    for feature_list in list_chunked_output[2]:
        insert_features_dict['citation_check'].extend(feature_list)
    for feature_list in list_chunked_output[3]:
        insert_features_dict['node_degree'].extend(feature_list)
    for feature_list in list_chunked_output[4]:
        insert_features_dict['reverse_max_sim'].extend(feature_list)


    # Compute other features per node pair
    for i,triple in enumerate(set_to_use):

        # Read basic data about the current node pair
        source = triple[0]
        target = triple[1]
        index_source = node_dict[source]
        index_target = node_dict[target]

        source_info = node_info[index_source]
        target_info = node_info[index_target]

        source_title = source_info[2].lower().split(" ")
        source_title = [token for token in source_title if token not in stpwds]
        source_title = [stemmer.stem(token) for token in source_title]

        target_title = target_info[2].lower().split(" ")
        target_title = [token for token in target_title if token not in stpwds]
        target_title = [stemmer.stem(token) for token in target_title]

        source_auth = source_info[3].split(",")
        target_auth = target_info[3].split(",") 

        # Creating features
        overlap_title = len(set(source_title).intersection(set(target_title)))
        insert_features_dict["overlap_title"].append(overlap_title)
        temp_diff = int(source_info[1]) - int(target_info[1])
        insert_features_dict["temp_diff"].append(temp_diff)
        comm_auth = len(set(source_auth).intersection(set(target_auth)))
        insert_features_dict["comm_auth"].append(comm_auth)

        peer_pop = peer_popularity(graph,source,target)
        insert_features_dict["peer_popularity"].append(peer_pop)

        LSA_dist = LSA_distance(source,target,node_dict,l)
        insert_features_dict["LSA_distance"].append(LSA_dist)

        title_weighted = t_titles.getrow(index_source).dot(t_titles.getrow(index_target).transpose())[0,0]
        insert_features_dict["title_sim"].append(title_weighted)

        N_LSA_dist = LSA_distance(source,target,node_dict,l_ngrams)
        insert_features_dict["N_LSA_distance"].append(N_LSA_dist)

        temporal_fit = temp_fit(source,target,graph,node_dict,publication_years)
        insert_features_dict["temporal_fit"].append(temporal_fit)

        path_length_value = path_length(source, target, all_path_dict)
        insert_features_dict["path_length"].append(path_length_value)

        if i%1000==0:
            print(i,"/",len(set_to_use))

    # Reshape features into np column arrays, one row per node pair
    for (name,value) in insert_features_dict.items():
        print(name,len(value))
        insert_features_dict[name] = prep.to_feature_shape(value)
    
    return insert_features_dict
