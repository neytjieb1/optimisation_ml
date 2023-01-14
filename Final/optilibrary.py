def importData(p=0.2):
    from sklearn import datasets
    dataset = datasets.fetch_california_housing(as_frame = True)
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import numpy as np
    np.random.seed(1)

    dataset.frame_normalized = StandardScaler().fit_transform(dataset.frame)
    # We drop Longitude as well since Latitude has enough information
    X = dataset.frame_normalized[:,0:-1]
    y = dataset.frame_normalized[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = p, random_state = 16)
    X_train = np.insert(X_train, 0, np.ones(X_train.shape[0]), axis=1)
    X_test = np.insert(X_test, 0, np.ones(X_test.shape[0]), axis=1)

    return X_train, y_train, X_test, y_test