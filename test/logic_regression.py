import numpy as np
from scipy.optimize import minimize

class MyLogisticRegression:
    """
    对于逻辑回归的理解：
    1、特征值X（[m, n]，m行样本值，n列特征），标签值y（有限的枚举值，[m, 1]，对应m行样本值的分类，分类只有1列）;
    2、逻辑回归的本质：找到特征值X与标签值y的映射关系（线性回归）; 接着，再通过sigmoid函数，将X到y的线性回归的输出，转换为[0, 1]之间的值;
        1）首先，线性回归本身就包含了一个梯度下降，sigmoid之后也包含一个梯度下降
        2）scipy.optimize.minimize方法，如何应用到预测方法中，以实现梯度下降的效果
    """
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        """
        训练方法
        :return:
        """
        pass

    def predict(self):
        """
        预测方法
        :return:
        """
        pass