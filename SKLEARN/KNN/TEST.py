import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def get_data():
    random_data = np.random.random((4, 3, 2))
    return random_data
def f(x, k, b):
    return k*x + b
if __name__ == "__main__":
    test_data = get_data()

    X = test_data[:,0]
    y = test_data[:]

    print(X)
    print("-**-*-*-*-*-*-*-*-")
    print(y)

    X = test_data[:, 0, 0]
    y = test_data[:]

    print(X)
    print("-**-*-*-*-*-*-*-*-")
    print(y)