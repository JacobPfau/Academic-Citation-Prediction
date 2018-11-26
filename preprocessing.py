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
