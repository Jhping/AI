"""
list_1 = ["j", "h", "p"]

str_1 = "".join(list_1)
print(str_1)
"""
import numpy as np
Y_predict = [1.1,0.2,3,0.8]
Y_predict = [1 if m > 0.5 else 0 for m in Y_predict]
print (Y_predict)
