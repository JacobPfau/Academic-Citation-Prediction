from sklearn.feature_extraction.text import TfidfVectorizer as Vectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KDTree
import numpy as np

##############################################################
#LSA

def LSA(tfidf,n_components=200):


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

def KDtree(features):
	return KDTree(features_LSA)

##############################################################
#Node degree
"""
@returns: length 2 array containing IN degree and OUT degree
"""

def node_degree(node_ID,graph):
	return np.array([graph.degree(node_ID,mode='IN'),graph.degree(node_ID,mode='OUT')])