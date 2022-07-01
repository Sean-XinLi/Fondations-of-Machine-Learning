# -*- coding: utf-8 -*-
"""
File:   assignment01.py
Author: Xin Li
Date:   09/19/2021
Desc:   Object-oriented design concept

"""

""" =======================  Import dependencies ========================== """
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.close('all')  # close any open plots

""" ======================  Function definitions ========================== """


def plotData(x1, t1, x2=None, t2=None, x3=None, t3=None, x4=None, t4=None, legend=[]):
    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the
       training data, the true function, and the estimated function'''
    p1 = plt.plot(x1, t1, 'bo')  # plot test data
    if (x2 is not None):
        p2 = plt.plot(x2, t2, 'm')  # plot training data -evenly spaced centers
    if (x3 is not None):
        p3 = plt.plot(x3, t3, 'g')  # plot true value -data centers
    if (x4 is not None):
        p4 = plt.plot(x4, t4, 'r')  # plot training data

    # add title, legend and axes labels
    plt.ylabel('t')  # label x and y axes
    plt.xlabel('x')

    if (x2 is None):
        a = plt.legend((p1[0]), legend, loc=4, fontsize=12)
    if (x3 is None):
        a = plt.legend((p1[0], p2[0]), legend, loc=4, fontsize=12)
    if (x4 is None):
        a = plt.legend((p1[0], p2[0], p3[0]), legend, loc=4, fontsize=12)
    else:
        a = plt.legend((p1[0], p2[0], p3[0], p4[0]), legend, loc=4, fontsize=12)


class RbfModule(object):
    '''
    this class is for learning RBF module in assignment01 of FML
    all the function in this class use to satisfy request of Question
    include:
    1、two method of setting means, evenly spaced centers and randomly sampled centers.
    2、create RBF module
    3、predict value based on test data
    4、calculate the true value
    5、calculate an absolute error value

    you can assigned evenly or randomly in method1 to choose optimal way to set means
    if want to use true_value, abs_error,function, must assign test data in x_test
    '''

    def __init__(self, m, s, bonds, x_train, t_train, x_test=None, method1="evenly"):

        # init, load the data and data preprocessing
        self.m = m
        self.s = s
        self.x_axis_bond_bottom = bonds[0]
        self.x_axis_bond_top = bonds[1]
        self.x_train = x_train
        self.t_train = t_train
        self.x_train_matrix = x_train.reshape(-1, 1)
        self.t_train_matrix = t_train.reshape(-1, 1)
        self.x_test = x_test
        self.method1 = method1

    # solution 1 -- set mean
    def set_mean_evenly(self):
        """
            this function divide the range of x values into M parts evenly
            and each part mean is the middle of each phase

            parameters:
            m is the number dimension of feature vector
            bond_low, bond_up represent bond of range of x values respectively

            return:
            means is a numpy array of all the values of mean -- center points

            """
        m = self.m

        # python role of function range is [)
        x_range = range(self.x_axis_bond_bottom, self.x_axis_bond_top + 1)
        x_len = (max(x_range) - min(x_range))

        # find the far left mean value
        u_init = x_len / (2 * m) - 4
        u = []
        u.append(u_init)

        for i in range(m - 1):
            # set all the values of means
            u_new = u_init + x_len / m * (i + 1)
            u.append(u_new)

        self.means = np.array(u)

        return self.means

    # solution 2 - - set mean
    def set_mean_randomly(self):
        """
        this function use each training points as mean values if M = N
        where N is the number of training data points
        if M < N, the function will randomly sample training data points
        to use as the mean values

        parameter:
        m, is the number dimension of feature vector
        x, is the training data points
        return:
        means, is a array of all the values of mean
        """

        # x1 is 2-dimensional matrix, need to translate to 1-dimensional
        m = self.m
        x = self.x_train

        # lock the random sequence
        # np.random.seed(0)

        u = np.random.choice(x, len(x), replace=False)
        for i in range(len(x) - m):
            u = np.delete(u, -1)

        self.means = np.array(u)
        a = np.ones(m)

        self.means.resize([m, ])

        return self.means

    # function with bias row
    def w_rbf(self):
        """
        this function is trying to find optimal w
        and add bias row to avoid the singular matrix occurring

        warn:
        this function RBF kernel maybe singular matrix using randomly space centers
        to avoid this situation occurring, use endless loop and try+except to work out

        parameter:
        u is means with np.array(m,)-style
        x is train data with np.array(m,1)-style
        s is variance of the exp

        return:
        w is the optimal coefficient of training model
        """
        while True:
            try:
                if self.method1 == 'evenly':
                    u = self.set_mean_evenly()
                else:
                    u = self.set_mean_randomly()
                x = self.x_train_matrix
                t = self.t_train_matrix
                s = self.s

                # add bias row as the first row of preprocessing data
                row = np.ones([1, 1])
                x1_new = np.row_stack((row, x))
                t1_new = np.row_stack((row, t))
                rbf_kernels = np.exp((-1) / (2 * s ** 2) * (x1_new - u) ** 2)
                self.w = np.linalg.inv(rbf_kernels.T @ rbf_kernels) @ rbf_kernels.T @ t1_new
                return self.w
            except np.linalg.LinAlgError as e:
                print("hold on, please")

    def rbf_training_model(self):
        """
        this function is a module applying for training data by using test data

        parameter:
        w is the optimal coefficient of training model
        x is test data
        m is a array of all the values of mean

        return:
        y is the values of the training test data
        """
        w = self.w_rbf()
        m = self.means
        x = self.x_test

        # data preprocessing
        x = x.reshape(-1, 1)

        self.y_test = w.T @ ((np.exp((-1) / (2 * s ** 2) * (x - m) ** 2)).T)

        # (1,7000) ===>(7000,)
        self.y_test = self.y_test.T.reshape(-1, )
        return self.y_test

    def true_value(self):
        """
        this function apply to calculate the ture value of f(x)
        f(x) is a function with variable x

        parameter:
        x is input data x

        return:
        x_new is ascending order of input data x
        true_value is the ture value
        """
        # data processing -- to robust the function
        x = self.x_test
        x_new = x.reshape(-1, )

        # x_new + 1 can not be zero
        x_new[x_new == -1] = -0.9
        # value of x is ascending order
        true_value = x_new / (1 + x_new)

        return x_new, true_value

    # get mean abs_error
    def abs_error(self):
        """
        this function apply to calculate the absolute error
        which means |y_predict - y_true|
        this function need to claim a method1 = "evenly" first

        :return:
        y_mean is average of the absolute error
        """
        m = self.m
        x_new, y_true = self.true_value()

        # train test data
        y_test = self.rbf_training_model()

        # data processing
        y_abs = abs(y_test - y_true)
        set_y_new = np.array(y_abs)

        y_mean = np.sum(set_y_new) / set_y_new.shape[0]
        return y_mean


def polynomial_func(x_train, t_train, x_test, M):
    '''
    this function use to get the predict values in polynomial method simply
    first, input train data and their ture values and a exponent of polynomial M to generate
    a bunch of optimal w
    second, input test value to get predict values

    :parameter:
    x_train is train data
    t_train is ture values of train data
    M is a exponent of polynomial
    x_test is test data

    :return:
    y_poly_predic, is predict value of test data
    '''
    # This needs to be filled in
    # training module
    X = np.array([x_train ** m for m in range(M + 1)]).T
    w = np.linalg.inv(X.T @ X) @ X.T @ t_train

    # generate predict values
    X_test = np.array([x_test ** m for m in range(w.size)]).T
    y_poly_predic = X_test @ w

    return y_poly_predic


def error_func(y_poly_predic, y_true):
    """
    this function uses to get absolute error
    :param y_predic: predict value of test data
    :param y_true: true value of test data
    :return:y_mean: a number about average of absolute error
    """
    y_abs = abs(y_poly_predic - y_true)
    set_y_new = np.array(y_abs)
    y_mean = np.mean(set_y_new)
    return y_mean


""" ======================  Variable Declaration ========================== """

# best M is 7 for set_mean_evently
# best M is 12 for set_mean_randomly
M = 5  # regression model order
s = 0.5
bonds = [-4, 4]

""" =======================  Load Training Data ======================= """

# load training data
data_uniform = np.load('train_data.npy')
data_uniform = data_uniform.T
x1 = data_uniform[:, 0]
t1 = data_uniform[:, 1]



""" ========================  Train the Model ============================= """
"""This is where you call functions to train your model with different RBF kernels   """
test1 = RbfModule(M, s, bonds, x1, t1, method1='evenly')
# get center points in one way
means = test1.set_mean_evenly()
# train RBF model and get the optimal coefficient of training model
w = test1.w_rbf()

""" ======================== Load Test Data  and Test the Model =========================== """

# load test data
x_test = np.load('test_data.npy')

# data preprocessing
x_test.resize(7000, 1)

# use different methods setting means to test the model
test2 = RbfModule(M, s, bonds, x1, t1, x_test, method1='evenly')
y_test1 = test2.rbf_training_model()
test3 = RbfModule(M, s, bonds, x1, t1, x_test, method1='randomly')
y_test2 = test3.rbf_training_model()
"""This is where you should load the testing data set. You shoud NOT re-train the model   """

""" ========================  Plot Results ============================== """
# draw the picture

# create plot 1-4

fig1 = plt.figure(figsize=([25, 25]), dpi=100)
plt.suptitle('M form 1 to 20, fixed S and different selected center values', fontsize=35)
legends = ['train', 'evenly space', 'randomly space', 'true value']
for i in range(1, 21):
    # use different methods setting means to test the model
    assign1 = RbfModule(i, s, bonds, x1, t1, x_test, method1='evenly')
    y_pred1 = assign1.rbf_training_model()
    assign2 = RbfModule(i, s, bonds, x1, t1, x_test, method1='randomly')
    y_pred2 = assign2.rbf_training_model()

    # input test data and get their ture value
    assign3 = RbfModule(i, s, bonds, x1, t1, x_test=x_test)
    x_order, y_true = assign3.true_value()

    plt.subplot(4, 5, i)
    plt.title('M={},S={}'.format(i, s), fontsize=18)
    plotData(x1, t1, x2=x_test, t2=y_pred1,
             x3=x_test, t3=y_pred2, x4=x_order, t4=y_true, legend=legends)
    plt.ylim((-10, 4))
    plt.grid(b='major')

plt.savefig('vary_M.png')
plt.show()


# data preparation
# input test data x_test and get average of absolute error -- |y_predict - y_true|
# using evenly spaced centers
# get the Question 5 data

# set_y1 = []
# set_m = []
# for i in range(1, 20 + 1):
#     assign4 = RbfModule(i, s, bonds, x1, t1, x_test=x_test, method1='evenly')
#     y_predict_mean = assign4.abs_error()
#     set_y1.append(y_predict_mean)
#     set_m.append(i)

# input test data x_test and get average of absolute error -- |y_predict - y_true|
# using randomly spaced centers
# get the mean and std value after running 10-times
# get the question 6 data

# set_y2_times = np.zeros(shape=(10, 20))
# for j in range(10):
#     set_y2 = []
#     set_m = []
#     for i in range(1, 20 + 1):
#         assign5 = RbfModule(i, s, bonds, x1, t1, x_test=x_test, method1="randomly")
#         y_mean1 = assign5.abs_error()
#         set_y2.append(y_mean1)
#         set_m.append(i)
#
#     # list ===> np.array
#     set_y2_array = np.array(set_y2).T
#
#     # add each column into set_y2_times
#     set_y2_times[j, ] = set_y2_array
#
# set_y2_mean = np.mean(set_y2_times, axis=0)
# set_y2_std = np.std(set_y2_times, axis=0)
#
# # create plot 5-6
# fig2 = plt.figure(figsize=[16, 9], dpi=100)
#
# plt.plot(set_m, set_y1, 'r', linestyle='--', marker='.', markersize=15, label='absolute error')
# plt.errorbar(x=set_m, y=set_y2_mean, yerr=set_y2_std, ecolor='g', elinewidth=3,
#              label='error bar of average absolute error')
# plt.xlabel('M', fontsize=18)
# plt.title('Compared two different selected center values', fontsize=20)
# plt.xticks(set_m)
# plt.yticks(set_m)
# plt.ylim((0, 20))
# plt.legend(fontsize=18)
# plt.grid(b='major')
#
# plt.savefig('error1.png')
# plt.show()


# vary s form 0.001 to 10, set M = 5 and the randomly selected center values
# draw the picture
#
# fig3 = plt.figure(figsize=([25, 25]), dpi=100)
# plt.suptitle('fixed M, S from 0.001 to 10, the randomly selected center values', fontsize=35)
# list_s = np.linspace(0.001, 10, 20)
# legends = ['train', 'randomly space', 'true value']
# for i in range(0, 20):
#     # use different methods setting means to test the model, s keep 3decimal places
#     assign7 = RbfModule(M, round(list_s[i], 3), bonds, x1, t1, x_test, method1='randomly')
#     y_pred7 = assign7.rbf_training_model()
#
#     # get ture values
#     assign8 = RbfModule(M, round(list_s[i], 3), bonds, x1, t1, x_test=x_test)
#     x_order, y_true = assign8.true_value()
#
#     plt.subplot(4, 5, i + 1)
#     plt.title('M={}，S={:.3f}'.format(M, list_s[i]), fontsize=18)
#     plotData(x1, t1, x2=x_test, t2=y_pred7, x3=x_order, t3=y_true, legend=legends)
#     plt.ylim((-10, 4))
#     plt.grid(b='major')
#
# plt.savefig('vary_S.png')
# plt.show()


# discussion 1
# difference among evently space, randomly space, ploynomial
# calculate the average absolute error -- |y_predict - y_true|

# inital set of x values and y values
list_x = []
list_y_poly = []
list_y_evenly = []
list_y_randomly = []

# get the each x value and each y value needed
# for i in range(1, 21):
#     # use different methods setting means to test the model
#     assign1 = RbfModule(i, s, bonds, x1, t1, x_test, method1='evenly')
#     y_pred1 = assign1.rbf_training_model()
#     assign2 = RbfModule(i, s, bonds, x1, t1, x_test, method1='randomly')
#     y_pred2 = assign2.rbf_training_model()
#
#     # input test data and get their ture value
#     assign3 = RbfModule(i, s, bonds, x1, t1, x_test=x_test)
#     x_order, y_true = assign3.true_value()
#
#     # polynomial training
#     y_poly_predic = polynomial_func(x1, t1, x_test, i)
#
#     # data processing
#     y_poly = error_func(y_poly_predic, y_true)
#     y_evenly = error_func(y_pred1, y_true)
#     y_randomly = error_func(y_pred2, y_true)
#     list_x.append(i)
#     list_y_poly.append(y_poly)
#     list_y_evenly.append(y_evenly)
#     list_y_randomly.append(y_randomly)

# draw the picture

# fig4 = plt.figure(figsize=([20, 8]), dpi=100)
# legends = ['evenly space', 'randomly space', 'polynomial']
# plt.title('difference among evently space, randomly space, ploynomial', fontsize=20)
# plotData(x1=list_x, t1=list_y_evenly, x2=list_x, t2=list_y_randomly, x3=list_x, t3=list_y_poly, legend=legends)
# plt.ylim((0, 20))
# plt.xticks(list_x)
# plt.grid(b='major')
#
# plt.savefig('error_2.png')
# plt.show()

""" This is where you should create the plots requested """
