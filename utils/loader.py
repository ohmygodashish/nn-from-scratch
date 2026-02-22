import numpy as np

def data_loader(data):
    """
    Helper function to load and split dataset into test and training sets.
    """
    data = np.array(data) # Convert df to numpy array

    m, n = data.shape
    #print(m, n)

    # Dataset splits
    test_size = m // 5
    train_size = m - test_size

    data_test = data[0:test_size].T
    data_train = data[test_size:m].T

    Y_test = data_test[0]
    X_test = data_test[1:n] / 255 # Scaled the pixel intensity values between 0 to 1
    Y_train = data_train[0]
    X_train = data_train[1:n] / 255 # Scaled the pixel intensity values between 0 to 1

    return X_train, Y_train, X_test, Y_test

