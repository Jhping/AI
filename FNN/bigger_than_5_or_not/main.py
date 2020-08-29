#!/usr/bin/python
import math
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import random
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25)
# reformulate the label.
# If the digit is smaller than 5, the label is 0.
# If the digit is larger than 5, the label is 1.
y_train[y_train < 5 ] = 0 #np.array支持，内建列表不支持
y_train[y_train >= 5] = 1
y_test[y_test < 5] = 0
y_test[y_test >= 5] = 1

##### 1 - Activation function
def sigmoid(z):
    return np.array([ 1/(1+math.exp((-1)*s)) for s in z])
    pass

##### 2  Initializaing parameters
def initialize_parameters(dim):# Random innitialize the parameters
    '''
        Argument: dim -- size of the w vector
        Returns:
        w -- initialized vector of shape (dim,1)
        b -- initializaed scalar
        '''
    w = np.array([random.random() for _ in range(int(dim))]).reshape(dim,1)
    b = random.random()
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w,b

#### 3 Forward and backward propagation
def propagate(w, b, X, Y):
    '''
    Implement the cost function and its gradient for the propagation

    Arguments:
    w - weights
    b - bias
    X - data
    Y - ground truth
    '''
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = np.sum(-1*np.dot(Y,np.log(2,A))-np.dot((1-Y),np.log(2,(1-A))))/m

    dw = np.dot(X,(A-Y).T)/m
    db = np.sum(A-Y)/m

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == (1))

    grads = {'dw': dw,
             'db': db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    '''
    This function optimize w and b by running a gradient descen algorithm

    Arguments:
    w - weights
    b - bias
    X - data
    Y - ground truth
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params - dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    '''

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads['dw']
        db = grads['db']

        w -= learning_rate*dw
        b -= learning_rate*db

        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights
    b -- bias
    X -- data

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T,X)+b)

    for i in range(A.shape[1]):
        if A[0][i] > 0.5:
            Y_prediction[0][i] = 1
        else:
            Y_prediction[0][i] = 0
    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost=True):
    """
    Build the logistic regression model by calling all the functions you have implemented.
    Arguments:
    X_train - training set
    Y_train - training label
    X_test - test set
    Y_test - test label
    num_iteration - hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d - dictionary should contain following information w,b,training_accuracy, test_accuracy,cost
    eg: d = {"w":w,
             "b":b,
             "training_accuracy": traing_accuracy,
             "test_accuracy":test_accuracy,
             "cost":cost}
    """
    d = {}
    dim = X_train.shape[1]
    w,b = initialize_parameters(dim)
    grads, cost = propagate(w, b, X_train, Y_train)
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=print_cost)
    Y_predict = predict(w, b, X_test)
    Y_predict = [1 if m > 0.5 else 0 for m in Y_predict]
    d["w"] = params["w"]
    d["b"] = params["b"]
    d["cost"] = costs
    d["training_accuracy"] = None
    n = 0
    for i in range(len(Y_predict)):
        if Y_predict[i] == Y_test[i]:
            n += 1
    d["test_accuracy"] = n/len(Y_predict)
    print(n/len(Y_predict))
    return d


def show_digits():
    print(digits.data.shape)#(1797, 64)
    print(digits.target.shape)#(1797,)
    for k, v in digits.items():
        print(k,v)

def show_img():
    for i in range(1,11):
        plt.subplot(2, 5, i)#图像的位置2X5的表格的第i个
        plt.imshow(digits.data[i - 1].reshape([8, 8]), cmap=plt.cm.gray_r)
        plt.text(3, 10, str(digits.target[i-1]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def show_train_data():
    print(X_train.shape)#(1347, 64)
    print(X_test.shape)#(450, 64)


if __name__ == "__main__":
    #show_digits()
    #show_img()
    #print(initialize_parameters(10))
    X_train=X_train.T
    Y_train=X_test.T
    X_test= y_train.T
    Y_test= y_test.T
    num_iterations=len(X_train )
    learning_rate = 0.01
    model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost=True)
    pass