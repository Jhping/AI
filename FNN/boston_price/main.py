#!/usr/bin/python
#-*- coding: utf8 -*-

import os,sys,time
import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample

class Node:
    # 该类为所有其他图节点类的父类
    def __init__(self, inputs=[]):
        # 定义每个节点的输入和输出
        self.inputs = inputs
        self.outputs = []

        # 每个节点都是其输入节点的输出节点
        for n in self.inputs:
            n.outputs.append(self)
            # set 'self' node as inbound_nodes's outbound_nodes

        self.value = None

        self.gradients = {}#对输入的梯度
        # keys are the inputs to this node, and their
        # values are the partials of this node with
        # respect to that input.
        # \partial{node}{input_i}
    def forward(self):# 计算自己的value
            # 前向传播函数 继承该类的其他类会覆写该函数
            '''
            Forward propagation.
            Compute the output value vased on 'inbound_nodes' and store the
            result in self.value
            '''

            raise NotImplemented

    def backward(self):# 计算自己对输入的梯度
            # 反向传播函数，继承该类的其他类会覆写该函数

            raise NotImplemented


class Input(Node):
    # 输入节点，包括神经网络输入节点X，权重节点W，和偏差节点B，该类节点没有输入节点
    def __init__(self):
        '''
        An Input node has no inbound nodes.
        So no need to pass anything to the Node instantiator.
        '''
        Node.__init__(self)

    def forward(self, value=None):
        '''
        Only input node is the node where the value may be passed
        as an argument to forward().
        All other node implementations should get the value of the
        previous node from self.inbound_nodes

        Example:
        val0: self.inbound_nodes[0].value
        '''
        # 定义节点数值
        if value is not None:
            self.value = value
            ## It's is input node, when need to forward, this node initiate self's value.

        # Input subclass just holds a value, such as a data feature or a model parameter(weight/bias)

    def backward(self):
        # 计算节点梯度
        self.gradients = {self: 0}  # initialization
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self] = grad_cost * 1#该类节点没有输入节点故*1

        # input N --> N1, N2
        # \partial L / \partial N
        # ==> \partial L / \partial N1 * \ partial N1 / \partial N


class Add(Node):
    def __init__(self, *nodes):
        Node.__init__(self, nodes)

    def forward(self):
        self.value = sum(map(lambda n: n.value, self.inputs))
        ## when execute forward, this node caculate value as defined.


class Linear(Node):
    # 全连接网络层的计算
    def __init__(self, nodes, weights, bias):#假设X = 1 x n，W = n x m, b = 1 x m 其中m为该层神经元的个数
        Node.__init__(self, [nodes, weights, bias])

    def forward(self):
        # 前向传播计算 y = w*x + b
        inputs = self.inputs[0].value
        weights = self.inputs[1].value
        bias = self.inputs[2].value

        self.value = np.dot(inputs, weights) + bias#计算的是 Z 的值 1 x m

    def backward(self):
        # 反向传播计算此节点对输入的梯度
        # initial a partial for each of the inbound_nodes.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}#zeros_like(a)获得一个和a相同shape的全 0 数组

        for n in self.outputs:
            # Get the partial of the cost w.r.t this node.
            grad_cost = n.gradients[self]
            # 以下分别计算对inputs， weights, bias的梯度
            self.gradients[self.inputs[0]] = np.dot(grad_cost, self.inputs[1].value.T)#(1xm)dot(mxn)=(1xn)
            self.gradients[self.inputs[1]] = np.dot(self.inputs[0].value.T, grad_cost)#(nx1)dot(1xm)=(nxm)
            self.gradients[self.inputs[2]] = np.sum(grad_cost, axis=0, keepdims=False)
        # WX + B / W ==> X
        # WX + B / X ==> W


class Sigmoid(Node):
    # 定义sigmoid函数
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1. / (1 + np.exp(-1 * x))

    def forward(self):
        # 前向 即为sigmoid函数计算
        self.x = self.inputs[0].value  # [0] input is a list
        self.value = self._sigmoid(self.x)

    def backward(self):
        # 反向传播计算梯度
        self.partial = self._sigmoid(self.x) * (1 - self._sigmoid(self.x))

        # y = 1 / (1 + e^-x)
        # y' = 1 / (1 + e^-x) (1 - 1 / (1 + e^-x))

        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}

        for n in self.outputs:
            grad_cost = n.gradients[self]  # Get the partial of the cost with respect to this node.

            self.gradients[self.inputs[0]] = grad_cost * self.partial
            # use * to keep all the dimension same!.


class MSE(Node):
    # 定义平均平方误差
    def __init__(self, y, a):
        Node.__init__(self, [y, a])

    def forward(self):
        # 前向传播计算
        y = self.inputs[0].value.reshape(-1, 1)#第二维元素个数需为1，第一维多少元素按情况定
        a = self.inputs[1].value.reshape(-1, 1)
        assert (y.shape == a.shape)

        self.m = self.inputs[0].value.shape[0]
        self.diff = y - a

        self.value = np.mean(self.diff ** 2)

    def backward(self):
        # 反向计算相应梯度
        self.gradients[self.inputs[0]] = (2 / self.m) * self.diff
        self.gradients[self.inputs[1]] = (-2 / self.m) * self.diff


def forward_and_backward(outputnode, graph):
    # execute all the forward method of sorted_nodes.

    ## In practice, it's common to feed in mutiple data example in each forward pass rather than just 1. Because the examples can be processed in parallel. The number of examples is called batch size.
    for n in graph:
        n.forward()
        ## each node execute forward, get self.value based on the topological sort result.

    for n in graph[::-1]:#从后向前取
        n.backward()

    # return outputnode.value


###   v -->  a -->  C
##    b --> C
##    b --> v -- a --> C
##    v --> v ---> a -- > C

def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.
    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.
    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outputs:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)
    """
    for k,v in G.items():
        print("************************************")
        print("node in G: %s" % k)
        print("n.in:%s" % v["in"])
        print("n.out:%s" % v["out"])
    """
    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]
            ## if n is Input Node, set n'value as
            ## feed_dict[n]
            ## else, n's value is caculate as its
            ## inbounds

        L.append(n)
        for m in n.outputs:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    for i in L:
        print("************************************")
        print("node in G: %s" % i)
        print("n.in:%s" % i.inputs)
        print("n.out:%s" % i.outputs)

    return L


def sgd_update(trainables, learning_rate=1e-2):
    # there are so many other update / optimization methods
    # such as Adam, Mom,
    for t in trainables:
        t.value += -1 * learning_rate * t.gradients[t]


if __name__ == "__main__":
    """
    Check out the new network architecture and dataset!
    Notice that the weights and biases are
    generated randomly.
    No need to change anything, but feel free to tweak
    to test your network, play around with the epochs, batch size, etc!
    """
    # from miniflow import *
    losses = []
    # Load data
    data = load_boston()
    X_ = data['data']       #506x13
    y_ = data['target']     #506x1
    # Normalize data
    X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)
    n_features = X_.shape[1]  # 13个特征值
    n_hidden = 10# 一个隐藏层有10个节点
    W1_ = np.random.randn(n_features, n_hidden) #13x10
    b1_ = np.zeros(n_hidden)                    #10
    W2_ = np.random.randn(n_hidden, 1)          #10x1
    b2_ = np.zeros(1)                           #1
    # Neural network
    X, y = Input(), Input()
    W1, b1 = Input(), Input()
    W2, b2 = Input(), Input()


    l1 = Linear(X, W1, b1)
    s1 = Sigmoid(l1)
    l2 = Linear(s1, W2, b2)
    cost = MSE(y, l2)  # 是L2不是十二

    feed_dict = {
        X: X_,#506x13
        y: y_,#506x1
        W1: W1_,#13x10
        b1: b1_,#10x1
        W2: W2_,#10x1
        b2: b2_#1x1
    }

    epochs = 5000
    # Total number of examples
    m = X_.shape[0]#506
    print("m:%s " % m)
    batch_size = 16
    steps_per_epoch = m // batch_size #31
    print("steps_per_epoch:%s " % steps_per_epoch)
    graph = topological_sort(feed_dict)
    #input("w...")
    trainables = [W1, b1, W2, b2]

    print("Total number of examples = {}".format(m))

    # Step 4
    for i in range(epochs):
        loss = 0
        for j in range(steps_per_epoch):
            # Step 1
            # Randomly sample a batch of examples
            X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

            # Reset value of X and y Inputs
            print("X_batch: %s  %s" % (len(X_batch),X_batch))
            print("y_batch: %s  %s" % (len(y_batch),y_batch))
            #input("waite")
            X.value = X_batch #16*13
            y.value = y_batch #16*1

            # Step 2
            _ = None
            forward_and_backward(_, graph)  # set output node not important.

            # Step 3
            rate = 1e-2

            sgd_update(trainables, rate)

            loss += graph[-1].value

        if i % 100 == 0:
            print("Epoch: {}, Loss: {:.3f}".format(i + 1, loss / steps_per_epoch))
            losses.append(loss)
