import features_pairwise as pw
import preprocessing as prep

class params:
    
    def  __init__(self,gold_graph, kdtree,l,node_dict,index_dict,pairs_subset_edges=True, 
                          chunk_size=1000,to_do = {'succ_pred':True, 'max_sim':True, 'citation_check':True, 
                                                   'node_degree':True, 'reverse_max_sim':True}, k_cc=500, metric_ms='COS', n_ms=3):
        self.gold_graph = gold_graph
        self.kdtree = kdtree
        self.l = l
        self.node_dict = node_dict
        self.index_dict = index_dict
        self.pairs_subset_edges = pairs_subset_edges
        self.chunk_size = chunk_size
        self.to_do = to_do
        self.k_cc = k_cc
        self.metric_ms = metric_ms
        self.n_ms = n_ms

    
    def by_chunk_noparams(self, pair_list):
        out = pw.by_chunk(pair_list,self.gold_graph,self.kdtree,self.l,self.node_dict,self.index_dict,
                         pairs_subset_edges=self.pairs_subset_edges, chunk_size = self.chunk_size,to_do = self.to_do, 
                          k_cc=self.k_cc, metric_ms=self.metric_ms, n_ms=self.n_ms)
        return out
    
    def all_paths_noparams(self, pairs_list):
        return prep.all_paths(pairs_list, self.gold_graph, pairs_subset_edges=self.pairs_subset_edges)
    
    def test(self, input_list):
        return input_list+2
