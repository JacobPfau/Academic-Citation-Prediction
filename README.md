This library predicts citations between physics papers given paper titles, authors, publication years, name of journal and abstract. We use a training set with 615,512 pairs of papers for which we know whether or not one cites the other. Our goal is then to predict given a set of 32,648 source/target pairs whether the source predicts the target. 

The first step in our pipeline was to construct a graph representing the citation relation between papers. Then we created features quantifying the relationship between each source/target pair. These features involved comparisons of abstract text and other paper descriptors and some used combinations of both text comparison and citation graph properties.

Following hyper-parameter tuning and feature selection, we determined that the library Light GBM — using a variant of gradient boosted trees — provides optimal citation prediction with yielding a ~0.980 f-score. 


## To reproduce our results run clean_notebook.ipynb
