import igraph
from sklearn.feature_extraction.text import TfidfVectorizer as Vectorizer
import numpy as np
import math
import copy
import random
from functools import reduce

# def succ_pred(pairs_array,gold_graph,pairs_subset_edges=True, chunk_size=1000):
#     num_chunks = math.ceil(len(pairs_array)/chunk_size)
#     l = len(pairs_array)
#     sp = []

#     for k in range(num_chunks):
#         chunk = pairs_array[ k*chunk_size : min([(k+1)*chunk_size,len(pairs_array)]) ]
#         chunk_removed_graph = copy.copy(gold_graph)

#         if pairs_subset_edges==True:
#             to_del = []
#             for triple in chunk:
#                 source = triple[0]
#                 target = triple[1]
#                 if triple[2]=='1':
#                     to_del.append((source,target))
#             chunk_removed_graph.delete_edges(to_del)

#         for triple in chunk:
#             source_ID = triple[0]
#             target_ID = triple[1]
#             succ_source = set(chunk_removed_graph.successors(source_ID))

#             pred_target = chunk_removed_graph.predecessors(target_ID)
#             pred_IDs = [chunk_removed_graph.vs[i].attributes()['name'] for i in pred_target if chunk_removed_graph.vs[i].attributes()['name']!=source_ID]
#             pred_succ = [set(chunk_removed_graph.successors(id)) for id in pred_IDs]

#             inter = [succ_source & p_s for p_s in pred_succ]
#             union = [succ_source | p_s for p_s in pred_succ]
#             len_union = [len(x) for x in union]
#             len_inter = [len(x) for x in inter]

#             jacc = [len_inter[i]/len_union[i] for i in range(len(len_union))]
#             if len(inter)>0:
#                 total_inter = len(reduce(lambda x,y:x|y,inter))
#             else:
#                 total_inter = 0
#             if len(len_inter)>0:
#                 stats = [max(len_inter),np.mean(len_inter),total_inter,max(jacc)]
#             else:
#                 stats = [0,0,total_inter,0]

#             sp.append(stats)

#         print(k, '/', num_chunks)
            
#     return sp

def all_paths(pairs_array, gold_graph, pairs_subset_edges=True, chunk_size=350):
    '''
    @pairs_subset_edges: indicates whether the pairs in pairs_array were used to construct the gold_graph. (i.e. usually true for training set, and false for competition set)
    @pairs_array: must be a list of lists (NOT np arrays)
    '''
    random.shuffle(pairs_array)
    num_chunks = math.ceil(len(pairs_array)/chunk_size)
    l = len(pairs_array)
    sources = set([t[0] for t in pairs_array])
    shortest_paths_dict = {s: dict() for s in sources}

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

        sources = list(set([t[0] for t in chunk]))
        targets = list(set([t[1] for t in chunk]))
        shortest_paths_array = np.array(chunk_removed_graph.shortest_paths_dijkstra(source = sources, target = targets, mode = 'OUT'))

        extension_dict = dict(zip( [s for s in sources],[dict(zip( [t for t in targets], row)) for row in shortest_paths_array] ))

        for pair in chunk:
            shortest_paths_dict[pair[0]][pair[1]] = extension_dict[pair[0]][pair[1]]

        if k%20==0:
            print(k, '/', num_chunks)

    return shortest_paths_dict


##############################################################
#tfidf

def tfidf(corpus, r= (1,1), midf = 3, madf=0.7,feats=10000, sublinear = True):
	vectorizer = Vectorizer(min_df = midf, max_df=madf, ngram_range=r,max_features=feats, stop_words='english', sublinear_tf = sublinear)
	X = vectorizer.fit_transform(corpus)

	print ("n_samples: %d, n_features: %d" % X.shape)

	return X

##############################################################
#node dictionary

def to_dict(k,v):
	"""
	zips keys, k, with values, v into dict
	"""
	return dict(zip( k,v ))

##############################################################
#Create igraph (article -> article)
#Input: IDs are supposed to be a list of the nodes IDs
#       edges should be a list of node ID tuples (ID_1,ID_2)
def article_graph(node_IDs, edges):
    ## create empty directed graph
    g = igraph.Graph(directed=True)
 
    ## add vertices
    g.add_vertices(node_IDs)
 
    ## add edges
    g.add_edges(edges)
    
    return g



# Computes |journals| x |journals| matrices (rows = sources)
# which give features about how papers from journals cite each other
# Input definition:
# journals: list, the journal column from node_info.
# journal_dict: dictionary mapping journal names to integers
# node_dict mapping article ID to it's index.
# train_edges 0 or 1 edges with 3 fields (source ID, target ID, 0/1)

# Return: [journal_dict,journal_features_matrix]
def journal_features(journals,node_dict,train_set,journal_dict=None):
    num_journals = 0
    # create journal_dict if not passed
    if journal_dict == None:
        journal_dict = dict()
        for i in range(len(journals)):
            if journals[i] not in journal_dict:
                journal_dict[journals[i]] = num_journals
                num_journals += 1
    
    num_journals = len(journal_dict.keys())
    count_ones = np.zeros((num_journals,num_journals))
    count_zeros = np.zeros((num_journals,num_journals))
    
    # count citations and 'not-citations'
    i = 0
    for edge in train_set:
        source_index = node_dict[edge[0]]
        target_index = node_dict[edge[1]]
        source_journal_id = journal_dict[journals[source_index]]
        target_journal_id = journal_dict[journals[target_index]]
        if int(edge[2]) == 0:
            count_zeros[source_journal_id,target_journal_id] += 1
        else:
            count_ones[source_journal_id,target_journal_id] += 1
        if i % 10000 == 0:
            print(i," training instances seen")
        i += 1
    
    
    
    # 2. also include 'non citations'
    # cell wise probabilites
    default_prob = 0.5
    cell_wise_matrix = np.ones((num_journals,num_journals))*default_prob
    for j_source in range(num_journals):
        for j_target in range(num_journals):
            num_edges = (count_ones[j_source,j_target] +
                        count_zeros[j_source,j_target])
            if( num_edges == 0 ):
                continue
            
            cell_wise_matrix[j_source,j_target] = count_ones[j_source,j_target] / num_edges
    
    # 1. only care about 'real citations' (edge[2] = 1)
    # How do citations from a journal spread to the other journals?
    # row wise probabilites
    citation_spread = count_ones
    for j_source in range(num_journals):
        if sum(citation_spread[j_source,:]) != 0:
            citation_spread[j_source,:] /= sum(citation_spread[j_source,:])
    
    journal_features = np.zeros((num_journals,num_journals,2))
    journal_features[:,:,0] = cell_wise_matrix
    journal_features[:,:,1] = citation_spread
            
    return [journal_dict,journal_features]
