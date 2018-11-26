from sklearn.feature_extraction.text import TfidfVectorizer as Vectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KDTree

##############################################################
#LSA

def LSA(tfidf,n_components=100):
Staging a file in Git’s terminology means adding it to the staging area, in preparation for a commit.


	print("Performing dimensionality reduction using LSA")

	svd = TruncatedSVD(n_components)
	normalizer = Normalizer(copy=False)
	lsa = make_pipeline(svd, normalizer)

	features_LSA = lsa.fit_transform(tfidf)

	explained_variance = svd.explained_variance_ratio_.sum()
	print("Explained variance of the SVD step: {}%".format(
	    int(explained_variance * 100)))

	return features_LSA

##############################################################
#KDtree

def KDtree(features)
	return KDTree(features_LSA)

