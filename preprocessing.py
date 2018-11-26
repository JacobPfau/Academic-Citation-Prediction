##############################################################
#tfidf

def tfidf(corpus,df=0.5,feats=10000:
	vectorizer = Vectorizer(max_df=df, max_features=feats, stop_words='english')
	X = vectorizer.fit_transform(corpus)

	print ("n_samples: %d, n_features: %d" % X.shape)

    return X
