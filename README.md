# MiniProject1
MiniProject 1 Assignment for the course COMP 551 Applied Machine Learning at McGill University
- Julien Verecken
- Srikanth Amudala
- Kamal Maanicshah

The project is structured in the following manner :
- The two notebooks regroups data analysis on each datasets, it outputs ``clean_redwine.csv`` and ``clean_breastcancer.csv`` respectively in the folders winequality and breastcancer. Those are used for the next tasks.
- ``model.py`` contains both _logistic regression_ and _LDA_ models as well as a _k-fold_ method.
- ``test_script.py`` includes a run test for both models on a chosen dataset, with training on a training set and validation on a validation set.
- ``kfold_script.py`` includes a run test for both models on a chosen dataset, with k-fold cross validation.
- ``important_features.npy`` contains a sorted list of important interaction terms of the features of the _redwine_ dataset, by L1 regularization. More information on the method used is in the report.
- ``utils.py`` contains preprocessing methods for the data normalization.
- ``scikit_test.py`` has the sole purpose of providing a idea of the performance of other implementation of the linear models.
- Any other file or folder has no important relevance for the reader but only for the authors.
