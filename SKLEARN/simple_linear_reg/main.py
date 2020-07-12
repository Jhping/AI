#!/usr/bin/python
"""
最小二乘法线性回归：sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False,copy_X=True, n_jobs=1)
参数：
1、fit_intercept：boolean,optional,default True。是否计算截距，默认为计算。如果使用中心化(均值为0)的数据，可以考虑设置为False,
不考虑截距。注意这里是考虑，一般还是要考虑截距。
2、normalize：boolean,optional,default False。标准化（标准差为1）开关，默认关闭；该参数在fit_intercept设置为False时自动忽略。如果为
True,回归会标准化输入参数：(X-X均值)/||X||，当然啦，在这里还是建议将标准化的工作放在训练模型之前；若为False，在训练
模型前，可使用sklearn.preprocessing.StandardScaler进行标准化处理。
3、copy_X：boolean,optional,default True。默认为True, 否则X会被改写。
4、n_jobs：int,optional,default 1int。默认为1.当-1时默认使用全部CPUs ??(这个参数有待尝试)。
属性：
coef_：array,shape(n_features, ) or (n_targets, n_features)。回归系数(斜率)。---> k
intercept_: 截距  ---> b
方法：
1、fit(X,y,sample_weight=None)
X:array, 稀疏矩阵 [n_samples,n_features]
y:array [n_samples, n_targets]
sample_weight:array [n_samples]，每条测试数据的权重，同样以矩阵方式传入（在版本0.17后添加了sample_weight）。
2、predict(x):预测方法，将返回值y_pred
3、get_params(deep=True)： 返回对regressor 的设置值
4、score(X,y,sample_weight=None)：评分函数，将返回一个小于1的得分，可能会小于0
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def get_data():
    random_data = np.random.random((20, 2))
    return random_data
def f(x, k, b):
    return k*x + b
if __name__ == "__main__":
    test_data = get_data()
    X = test_data[:,0]
    y = test_data[:,1]
    reg = LinearRegression().fit(X.reshape(-1, 1), y)
    print("reg.coef_ :%s " % reg.coef_)
    print("reg.intercept_ :%s " % reg.intercept_)
    print("reg.score :%s " % reg.score(X.reshape(-1, 1), y))

    print("fit params :{}".format(reg.get_params(deep=True)))
    #show raw data and modue line
    plt.scatter(X, y)
    plt.plot(X, f(X, reg.coef_, reg.intercept_),color = "red")
    plt.scatter(0.1, reg.predict([[0.1]]), color = "black")
    plt.show()



