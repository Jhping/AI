#!/usr/bin/python

"""
KNeighborsClassifier函数
使用KNeighborsClassifier创建K临近分类器：

sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30,
             p=2, metric=’minkowski’, metric_params=None, n_jobs=None, **kwargs)
参数注释：
1，n_neighbors
临近的节点数量，默认值是5

2，weights
权重，默认值是uniform，
uniform：表示每个数据点的权重是相同的；
distance：离一个簇中心越近的点，权重越高；
callable：用户定义的函数，用于表示每个数据点的权重

3，algorithm
auto：根据值选择最合适的算法
ball_tree：使用BallTree
kd_tree：KDTree
brute：使用Brute-Force查找

4，leaf_size
leaf_size传递给BallTree或者KDTree，表示构造树的大小，用于影响模型构建的速度和树需要的内存数量，最佳值是根据数据来确定的，默认值是30。

5，p，metric，metric_paras
p参数用于设置Minkowski 距离的Power参数，当p=1时，等价于manhattan距离；当p=2等价于euclidean距离，当p>2时，就是Minkowski 距离。
metric参数：设置计算距离的方法
metric_paras：传递给计算距离方法的参数
6，n_jobs

并发执行的job数量，用于查找邻近的数据点。默认值1，选取-1占据CPU比重会减小，但运行速度也会变慢，所有的core都会运行。


1，kNN算法的计算步骤

kNN算法就是根据距离待分类样本A最近的k个样本数据的分类来预测A可能属于的类别，基本的计算步骤如下：

对数据进行标准化，通常是进行归一化，避免量纲对计算距离的影响；
计算待分类数据与训练集中每一个样本之间的距离；
找出与待分类样本距离最近的k个样本；
观测这k个样本的分类情况；
把出现次数最多的类别作为待分类数据的类别。
2，kNN算法如何计算距离？

在计算距离之前，需要对每个数值属性进行规范化，这有助于避免较大初始值域的属性比具有较小初始值域的属性的权重过大。

对于数值属性，kNN算法使用距离公式来计算任意两个样本数据之间的距离。
对于标称属性（如类别），kNN算法使用比较法，当两个样本数据相等时，距离为0；当两个样本数据不等时，距离是1。
对于缺失值，通常取最大的差值，假设每个属性都已经映射到[0,1]区间，对于标称属性，设置差值为1；对于数值属性，如果两个元组都缺失值，那么设置差值为1；
如果只有一个值缺失，另一个规范化的值是v，则取差值为 1-v 和 v 的较大者。
3，kNN算法如何确定k的值？

k的最优值，需要通过实验来确定。从k=1开始，使用检验数据集来估计分类器的错误率。重复该过程，每次k增加1，允许增加一个近邻，选取产生最小错误率的k。
一般而言，训练数据集越多，k的值越大，使得分类预测可以基于训练数据集的较大比例。在应用中，一般选择较小k并且k是奇数。通常采用交叉验证的方法来选取合适的k值。

数据点之间的距离，计算距离的方法有："euclidean"（欧氏距离）,”minkowski”（明科夫斯基距离）, "maximum"（切比雪夫距离）, "manhattan"（绝对值距离）,"canberra"（兰式距离）, 或 "minkowski"（马氏距离）等

"""




from sklearn.datasets import load_iris
iris_dataset=load_iris()
def show_iris_dataset():
    print(iris_dataset.target[0:4])
    print(iris_dataset.data[0:4])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(iris_dataset['data'], iris_dataset['target'],random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train, y_train)
assess_model_socre=knn.score(x_test,y_test)
print('Test set score:{:2f}'.format(assess_model_socre))

if __name__ == "__main__":
    show_iris_dataset()

    x_new = [[1.0,2.0,3.0,4.0]]
    prediction = knn.predict(x_new)
    print("prediction :{0}  ,classifier:{1}".format(prediction, iris_dataset["target_names"][prediction]))
    pass