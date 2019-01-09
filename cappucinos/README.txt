Submission to the INF554 Project
Team: cappucinos

FILESYTEM:
We expect the files to be just as in our submission, i.e.

someFolder
├── data
│   ├── node_information.csv
│   ├── testing_set.txt
│   └── training_set.txt
├── construction_notebook.ipynb
├── features_nodewise.py
├── features_pairwise.py
├── lightgbm_model
├── multi_func.py
├── pickled_notebook.ipynb
└── preprocessing.py


HOW TO GET PREDICTIONS? (e.g. for the private testing set)?
1. Replace the testing_set.txt file with your file
2. Run all cells in the pickled_notebook.ipynb Jupyter Notebook
3. A new file "predictions.csv" will be created


Why do we load a pickled model?
Computing the our features on the whole training set took 3 hours on a 24 core Google cloud server. To accelerate the evaluation we included the pretrained model in our submission. The feature computation on the test set will still take some time.
In case you also want to reconstruct the model from scratch, then you can run the construction_notebook.ipynb. The only difference from construction_notebook.ipynb and pickled_notebook.ipynb is that construction_notebook.ipynb also builds and trains the model whereas pickled_notebook skips the training step by loading the model from disk.
