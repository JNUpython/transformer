# -*- coding: utf-8 -*-
# @Time    : 2019/6/25 21:52
# @Author  : kean
# @Email   : ?
# @File    : temp.py
# @Software: PyCharm


import tensorflow as tf
import numpy as np
from utils import logger

tf.InteractiveSession()

"""
测试tf.py_func函数的作用
"""


def func(inpt):
    # 将tensor 作为narray进行操作增加 灵活性
    print(type(inpt))
    return inpt * 10


xarray = np.arange(10)
xtensor = tf.constant(value=xarray, dtype=tf.int32)
res = tf.py_func(func, [xtensor], tf.int32)
logger.info(type(res))
logger.info(res.eval())
