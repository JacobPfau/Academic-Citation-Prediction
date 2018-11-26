from sklearn.feature_extraction.text import TfidfVectorizer as Vectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD

##############################################################
#LSA

def LSA(corpus,max_df=0.5,max_features=10000,n_components=100):

	vectorizer = Vectorizer(max_df=0.5, max_features=10000, stop_words='english')
	X = vectorizer.fit_transform(corpus)

	print ("n_samples: %d, n_features: %d" % X.shape)

	n_components = 100

	print("Performing dimensionality reduction using LSA")

	svd = TruncatedSVD(n_components)
	normalizer = Normalizer(copy=False)
	lsa = make_pipeline(svd, normalizer)

	features_LSA = lsa.fit_transform(X)

	explained_variance = svd.explained_variance_ratio_.sum()
	print("Explained variance of the SVD step: {}%".format(
	    int(explained_variance * 100)))

	return features_LSA

##############################################################
