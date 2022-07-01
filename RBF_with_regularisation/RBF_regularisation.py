# -*- coding: utf-8 -*-
"""
File:   assignment02.py
Author: Xin Li
Date:
Desc:
"""

""" =======================  Import dependencies ========================== """
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

plt.close('all')  # close any open plots

""" ======================  Function definitions ========================== """


def plotData(x1, t1, x2=None, t2=None, x3=None, t3=None, legend=[]):
    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the
       training data, the true function, and the estimated function'''
    p1 = plt.plot(x1, t1, 'bo', linestyle=' ', markersize=4)
    if (x2 is not None):
        p2 = plt.plot(x2, t2, 'm', linewidth='3')  # plot training data
    if (x3 is not None):
        p3 = plt.plot(x3, t3, 'g', linewidth='3')  # plot true value -

    # add title, legend and axes labels
    plt.ylabel('t')  # label x and y axes
    plt.xlabel('x')

    if (x2 is None):
        a = plt.legend((p1[0]), legend, loc=4, fontsize=12)
    if (x3 is None):
        a = plt.legend((p1[0], p2[0]), legend, loc=4, fontsize=12)
    else:
        a = plt.legend((p1[0], p2[0], p3[0]), legend, loc=4, fontsize=12)


class Regression(object):
    '''
        this class is used to find optimal w with gradient descent
        this class need assign iteration time of gradient descent, learning
        rate.
        :parameter:
        n_iterations, iteration time of gradient descent
        learning_rate, learning rate
        '''

    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initialize_w(self, x_train, t_value):
        obj1 = RadialBasisFunction(s)
        self.w = obj1.find_w(x_train, t_value)
        return self.w

    def fit(self, x, t_value):
        # Insert constant ones for bias weights
        x_new = np.insert(x, 0, 1)
        t = np.insert(t_value, 0, 1)

        # data preprocessing
        t.resize(len(t), 1)

        # get the initialize w -- self.w
        self.w = self.initialize_w(x, t_value)
        # E(w)_rbf =  1/2 * (y(x,w) - t) ^2
        # derivative E(w)_rbf
        obj2 = RadialBasisFunction(s)
        rbf_kernels = obj2.kernels(x, x)
        # do gradient descent for n_iterations
        for i in range(self.n_iterations):
            y_predict = self.predict1(self.w, x, x)

            # gradient rate equal the gradient rate of E(w) + regularization
            rbf_grad = rbf_kernels.T @ rbf_kernels @ self.w - rbf_kernels.T @ t

            obj3 = L1_L2_regularization(alpha=alpha, l1_ratio=l1_ratio)
            grad_w = rbf_grad + obj3.grad(self.w)
            self.w -= self.learning_rate * grad_w

        return self.w

    def predict1(self, w, x_train, x_test):
        obj4 = RadialBasisFunction(s)
        rbk_kernels = obj4.kernels(x_train, x_test)

        y_predict = w.T @ rbk_kernels.T
        y_predict = y_predict.reshape(-1, 1)
        return y_predict


class RadialBasisFunction(object):
    """
    this class is used to create w for initialization w in class regression
    the way setting means is randomly sampled centers.
    the whole means is training data points
    :parameter
    x_train: train data
    t_value: target value of train data
    """

    def __init__(self, s):
        self.s = s

    def set_mean(self, x):
        self.m = x

    def kernels(self, x_train, x_test):
        '''
        calculate rbf kernels
        and add bias row to avoid the singular matrix occurring

        :param x_train:
        :return: ref_kernels
        '''
        self.set_mean(x_train)

        # insert constant ones for bias weights
        self.m = np.insert(self.m, 0, 1)
        self.x = np.insert(x_train, 0, 1)
        self.x_test = np.insert(x_test, 0, 1)

        # data preprocessing
        self.m.resize(1, len(self.m))
        self.x.resize(len(self.x), 1)
        self.x_test.resize((len(self.x_test)), 1)

        rbf_kernels = np.exp((-1) / (2 * self.s ** 2) * (self.x_test - self.m) ** 2)

        return rbf_kernels

    def find_w(self, x_train, t_value):
        """
        this function is trying to find optimal w
        and add bias row to avoid the singular matrix occurring

        """

        # insert constant ones for bias weights
        t = np.insert(t_value, 0, 1)

        # data preprocessing
        t.resize(len(t), 1)

        # calculate the rbf kernels
        rbf_kernels = self.kernels(x_train, x_train)

        self.w = np.linalg.inv(rbf_kernels.T @ rbf_kernels) @ rbf_kernels.T @ t

        return self.w


class L1_L2_regularization(object):
    '''
    regularization for Elastic Net Regression
    '''

    def __init__(self, alpha, l1_ratio):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def grad(self, w):
        l1_grad = self.l1_ratio * np.sign(w)
        l2_grad = (1 - self.l1_ratio) * 2 * w

        return self.alpha * (l1_grad + l2_grad)


class ElasticNet(Regression):
    '''
    Regression where a combination of l1 and l2 regularization are used.
    The ratio of their contributions are set with the 'l1_ratio' parameter.
    :parameter
    alpha：the facotr that will determine the amount of regularization
    and feature shrinkage.
    l1_ratio: weights the contribution of l1 and l2 regularization.
    n_iterations: the number of training iterations the algorithm will tune
    the weights for.
    learning_rate: the step length that will be used when updating the weights.
    '''

    def __init__(self, alpha, l1_ratio, n_iterations, learning_rate):
        self.regularization = L1_L2_regularization(alpha=alpha, l1_ratio=l1_ratio)
        super(ElasticNet, self).__init__(n_iterations, learning_rate)

    # fit
    def fit1(self, x_train, t_value):
        super(ElasticNet, self).fit(x_train, t_value)

    # predict
    def predict(self, x_train, t_value, x_test):
        self.fit1(x_train, t_value)

        return super(ElasticNet, self).predict1(self.w, x_train, x_test)


def true_value_function(x):
    return 3 * (x + np.sin(x)) * np.exp(-x ** 2.0)


def error_func(x_train, t_value, x_test, y_true):
    obj5 = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, n_iterations=1000, learning_rate=0.0001)
    y_predict = obj5.predict(x_train, t_value, x_test)

    # delete the first element
    y_predict = np.delete(y_predict, 0)
    abs_error = sum(abs(y_predict - y_true) / len(y_true))

    return abs_error


""" ======================  Variable Declaration ========================== """
a = None
while (a != "0"):
    s = 0.2
    l1_ratio = 0.9
    alpha = 0

    """ =======================  Load Training Data ======================= """

    # load training data
    data_set = np.load('data_set.npz')

    # get the data
    training_data = data_set['arr_0']
    validation = data_set['arr_1']
    test_data = data_set['arr_2']
    x_train = training_data[:, 0]
    t_value = training_data[:, 1]

    x_valid = validation[:, 0]
    t_valid = validation[:, 1]

    x_test_data = test_data[:, 0]
    t_test = test_data[:, 1]
    """ ========================  Plot Results ============================== """
    # get the true values
    y_train_true = true_value_function(x_train)
    y_valid_true = true_value_function(x_valid)
    y_test_true = true_value_function(x_test_data)

    # set parameter for 3D plot
    list_l1_ratio_3d = np.linspace(0, 1, 101)
    list_alpha_3d = np.linspace(0, 10, 11)
    error_coordinate = np.zeros((101, 11))

    print("What plot do you want to get? \n"
          "\n"
          "1==> Some performance of RBF using Elastic Net regularization with train data\n"
          "2==> 3-D abs_error on the training data\n"
          "3==> 3-D abs_error on the validation data\n"
          "4==> 3-D abs_error on the test data\n"
          "0==> exit")
    a = input("please type the number:")

    if a == "1":
        # draw the picture
        # create plot 1-3

        fig1 = plt.figure(figsize=([25, 25]), dpi=100)
        legends = ['train data', 'estimated value', 'true value']
        plt.suptitle('S=0.2, Alpha from 0 to 10, lambda from 0 to 1\n '
                     'some Performance of RBF using Elastic Net regularization with train data',
                     fontsize=30)

        list_l1_ratio = np.linspace(0, 1, 4)
        list_alpha = np.linspace(0, 10, 4)
        for i in range(16):
            row = i // 4
            col = i % 4
            l1_ratio = round(list_l1_ratio[row], 1)
            alpha = round(list_alpha[col], 0)
            assign1 = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, n_iterations=1000, learning_rate=0.01)

            y_predict = assign1.predict(x_train, t_value, x_train)

            # delete the first element
            y_predict = np.delete(y_predict, 0)

            plt.subplot(4, 4, i + 1)
            plt.title('alpha={},  l1_ratio={}'.format(alpha, l1_ratio), fontsize=18)
            plotData(x1=x_train, t1=t_value, x2=x_train, t2=y_predict, x3=x_train, t3=y_train_true, legend=legends)
            plt.ylim((-10, 10))
            plt.grid(b='major')
        plt.savefig('vary_alpha_and_lambda.png')
        plt.show()

    elif a == "2":

        # 3D_mesh -- absolute error on the training data across all combinations of alpha and l1_ratio
        # create plot 4

        for i in range(101 * 11):
            row = i // 11
            col = i % 11
            l1_ratio = np.round(list_l1_ratio_3d[row], 2)
            alpha = list_alpha_3d[col]
            error_coordinate[row, col] = error_func(x_train, t_value, x_train, y_train_true)

        fig2 = plt.figure(dpi=100)
        ax2 = plt.axes(projection='3d')
        list_alpha_3d, list_l1_ratio_3d = np.meshgrid(list_alpha_3d, list_l1_ratio_3d)

        surf2 = ax2.plot_surface(list_alpha_3d, list_l1_ratio_3d, error_coordinate,
                                 linewidth=0, alpha=0.5, cmap='jet')
        plt.colorbar(surf2, shrink=0.7)

        ax2.set(title='S=0.2, 3-D abs_error on the training data',
                xlabel='alpha', ylabel='l1 ratio', zlabel='abs_error')
        ax2.view_init(15, -135)
        plt.savefig('3D_error_training.png')
        plt.show()

    elif a == "3":
        # 3D_mesh -- absolute error on the validation data across all combinations of alpha and l1_ratio
        # create plot 5

        for i in range(101 * 11):
            row = i // 11
            col = i % 11
            l1_ratio = np.round(list_l1_ratio_3d[row], 2)
            alpha = list_alpha_3d[col]
            error_coordinate[row, col] = error_func(x_train, t_value, x_valid, y_valid_true)

        fig3 = plt.figure(dpi=100)
        ax3 = plt.axes(projection='3d')
        list_alpha_3d, list_l1_ratio_3d = np.meshgrid(list_alpha_3d, list_l1_ratio_3d)
        surf3 = ax3.plot_surface(list_alpha_3d, list_l1_ratio_3d, error_coordinate,
                                 linewidth=0, alpha=0.5, cmap='jet')

        plt.colorbar(surf3, shrink=0.7)
        ax3.set(title='S=0.2, 3-D abs_error on the validation data',
                xlabel='alpha', ylabel='l1 ratio', zlabel='abs_error')
        ax3.view_init(15, -135)
        plt.savefig('3D_error_validation.png')
        plt.show()

    elif a == "4":
        # 3D_mesh -- absolute error on the test data across all combinations of alpha and l1_ratio
        # create plot 6

        for i in range(101 * 11):
            row = i // 11
            col = i % 11
            l1_ratio = np.round(list_l1_ratio_3d[row], 2)
            alpha = list_alpha_3d[col]
            error_coordinate[row, col] = error_func(x_train, t_value, x_test_data, y_test_true)

        fig4 = plt.figure(dpi=100)
        ax4 = plt.axes(projection='3d')
        list_alpha_3d, list_l1_ratio_3d = np.meshgrid(list_alpha_3d, list_l1_ratio_3d)
        surf4 = ax4.plot_surface(list_alpha_3d, list_l1_ratio_3d, error_coordinate,
                                 linewidth=0, alpha=0.5, cmap='jet')
        plt.colorbar(surf4, shrink=0.7)
        ax4.set(title='S=0.2, 3-D abs_error on the test data',
                xlabel='alpha', ylabel='l1 ratio', zlabel='abs_error')
        ax4.view_init(15, -135)
        plt.savefig('3D_error_test.png')
        plt.show()
