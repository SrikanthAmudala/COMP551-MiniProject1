import numpy as np

"""
This package regroups methods to apply on the dataset before learning
"""

def preprocessing(df, prop=0.8):
    # Convert the dataframe to numpy array
    X_train = df.to_numpy()
    
    # Divide data set and normalize data separately (no information leakage)
    np.random.shuffle(X_train)
    isep = round(prop * X_train.shape[0])

    y_val = X_train[isep:, -1]
    y_train = X_train[:isep, -1]

    X_val = X_train[isep:, :-1]
    X_train = X_train[:isep, :-1]

    X_val = (X_val - np.mean(X_val, axis=0)) / np.std(X_val, axis=0)
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

    # Add bias term
    X_val = np.concatenate([np.ones((X_val.shape[0], 1)), X_val], axis=1)
    X_train = np.concatenate([np.ones((X_train.shape[0], 1)), X_train], axis=1)

    return (X_train, y_train, X_val, y_val)

def preprocessing_kfold(dataset_train, dataset_val):
    # Divide data set and normalize data separately (no information leakage)
    y_val = dataset_val[:, -1]
    y_train = dataset_train[:, -1]

    X_val = dataset_val[:, :-1]
    X_train = dataset_train[:, :-1]

    X_val = (X_val - np.mean(X_val, axis=0)) / np.std(X_val, axis=0)
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

    # Add bias term
    X_val = np.concatenate([np.ones((X_val.shape[0], 1)), X_val], axis=1)
    X_train = np.concatenate([np.ones((X_train.shape[0], 1)), X_train], axis=1)

    return (X_train, y_train, X_val, y_val)

def augment_square(df):
    dataset = df.copy()
    columns = dataset.columns[:-1]
    index = len(columns)
    for column in columns[::-1]:
        dataset.insert(index,column + '^2',np.square(dataset[column]))
    return dataset

def augment_interact(df):
    # Add all cross product and squares of features 
    dataset = df.copy()
    columns = dataset.columns[-2::-1]
    index = len(columns)
    for i in range(len(columns)):
        for j in range(i+1):
            dataset.insert(index,columns[i] + '*' + columns[j],dataset[[columns[i],columns[j]]].product(axis=1))
    
    return dataset
