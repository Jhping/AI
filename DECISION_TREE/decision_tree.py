#!/usr/bin/python

"""
信息熵：信源的平均不确定性应当为单个符号不确定性log(1/Pi)的统计平均值,
        即为sum{Pr(Xi)log2(1/Pr(Xi))}
Gini纯度：1-sum{Pr(i)}

1、取一个特征，根据特征值对对应结果分类，计算信息熵
2、依次计算每个特征，找出信息熵最小的特征值，作为优先决策项
3、根据上一次找出的最小信息熵提取特征数据，重复1和2.
"""
from icecream import ic
from collections import Counter
import numpy as np
import pandas as pd
mock_data = {
    'gender':['F', 'F', 'F', 'F', 'M', 'M', 'M'],
    'income': ['+10', '-10', '+10', '+10', '+10', '+10', '-10'],
    'family_number': [1, 1, 2, 1, 1, 1, 2],
   # 'pet': [1, 1, 1, 0, 0, 0, 1],
    'bought': [1, 1, 1, 0, 0, 0, 1],
}
def entropy(elements):
    '''群体的混乱程度'''
    counter = Counter(elements)
    probs = [counter[c] / len(elements) for c in set(elements)]
    ic(probs)
    return - sum(p * np.log(p) for p in probs)
if __name__ == "__main__":
    dataset = pd.DataFrame.from_dict(mock_data)
    print(dataset)
    # split_by_gender:
    print(entropy([1, 1, 1, 0]) + entropy([0, 0, 1]))
    # split_by_income:
    print(entropy([1, 1, 0, 0, 0]) + entropy([1, 1]))
    # split_by_family_number
    print(entropy([1, 1, 0, 0, 0]) + entropy([1, 1]))
    # 我们最希望找到一种feature， split_by_some_feature:
    # split_by_pet
    print(entropy([1, 1, 1, 1]) + entropy([0, 0, 0]) )