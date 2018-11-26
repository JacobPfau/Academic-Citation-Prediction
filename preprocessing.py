import igraph

##############################################################
#tfidf

def tfidf(corpus,df=0.5,feats=10000):
	vectorizer = Vectorizer(max_df=df, max_features=feats, stop_words='english')
	X = vectorizer.fit_transform(corpus)

	print ("n_samples: %d, n_features: %d" % X.shape)

    return X

##############################################################
#node dictionary

def to_dict(node_info):
	return dict(zip( [element[0] for element in node_info],range(len(node_info)) ))

##############################################################
#Create igraph (article -> article)
#Input: IDs are supposed to be a list of the nodes IDs
#       edges should be a list of node ID tuples (ID_1,ID_2)
def article_graph(node_IDs, edges):
    ## create empty directed graph
    g = igraph.Graph(directed=True)
 
    ## add vertices
    g.add_vertices(nodes)
 
    ## add edges
    g.add_edges(edges)
    
    return g
