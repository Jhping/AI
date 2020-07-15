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

注意事项
最小二乘的系数估计依赖于模型特征项的独立性。当特征项相关，并且设计矩阵X 的列近似线性相关时，
设计矩阵便接近于一个奇异矩阵，此时最小二乘估计对观测点中的随机误差变得高度敏感，产生较大方差。
例如，当没有试验设计的收集数据时，可能会出现这种多重共线性(multicollinearity )的情况。

思考（面试常问）
1.什么是线性分类模型，什么是非线性分类模型，它们各有什么有优缺点？

     区分是否为线性模型，主要是看一个乘法式子中自变量x前的系数w,如果w只影响一个x（注：应该是说x只被一个w影响），那么此模型为线性模型。或者判断决策边界是否是线性的。不满足线性模型的情况即为非线性模型。只考虑二类的情形，所谓线性分类器即用一个超平面将正负样本分离开，表达式为 y=wx 。这里是强调的是平面。而非线性的分类界面没有这个限制，可以是曲面，多个超平面的组合等。

    线性分类模型（LR,贝叶斯分类，单层感知机、线性回归）优缺点：算法简单和具有“学习”能力。线性分类器速度快、编程方便，但是可能拟合效果不会很好。

    非线性分类模型（决策树、RF、GBDT、多层感知机）优缺点：非线性分类器编程复杂，但是效果拟合能力强

2.线性回归和逻辑回归有什么区别和联系？

    ①区别：线性回归用来预测（如:房价预测），逻辑回归用来分类（如：疾病诊断）；线性回归是拟合函数，逻辑回归是预测函数；线性回归的参数计算方法是最小二乘法，逻辑回归的参数计算方法是梯度下降法。（附上吴恩达讲解）

    ②联系：逻辑回归比线性回归多了一个Sigmoid函数，使样本能映射到[0,1]之间的数值，用来做分类问题。

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



