#!/usr/bin/python

class NoNparrayError(BaseException):
    def __init__(self, value):
        self.value = value
    def __str__(self):# print(class)的输出结果
        return repr(self.value)


