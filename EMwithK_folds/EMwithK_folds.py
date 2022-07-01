import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
import time


def EM_GaussianMixture(train_data, test_data, NumberOfComponents, cov_type, c=1):
    # parameter
    MaximumNumberOfIterations = 100
    DiffThresh = 1e-4

    # data_processing
    train_data_order = train_data[np.argsort(train_data[:, 4])]
    train_data_seotsa = train_data_order[np.where(train_data_order == 'Iris-setosa')[0]]
    train_data_versicolor = train_data_order[np.where(train_data_order == 'Iris-versicolor')[0]]
    train_data_virginica = train_data_order[np.where(train_data_order == 'Iris-virginica')[0]]

    # data [0:150,0:5]
    X = train_data[0:150, 0:4]
    N = X.shape[0]  # number of data points
    d = X.shape[1]  # dimensionality
    rp = np.random.permutation(N)  # random permutation of numbers 1:N

    # Initialize Parameters
    Means = X[rp[0:NumberOfComponents], :]
    Sigs = np.zeros((d, d, NumberOfComponents))
    Ps = np.zeros((NumberOfComponents,))
    pZ_X = np.zeros((N, NumberOfComponents))

    for i in range(NumberOfComponents):
        Ps[i] = 1 / NumberOfComponents

    if cov_type == 'isotropic':
        for i in range(NumberOfComponents):
            Sigs[:, :, i] = c * np.eye(d)

    elif cov_type == 'full':
        for i in range(NumberOfComponents):
            while True:
                Sigs_plane = np.zeros((4, 4))
                Sigs_plane = np.random.random(Sigs_plane.shape)
                Sigs_plane = np.triu(Sigs_plane)
                Sigs_plane += Sigs_plane.T - np.diag(Sigs_plane.diagonal())
                Sigs_plane = Sigs_plane / np.sum(Sigs_plane)
                w, v = np.linalg.eig(Sigs_plane)
                if np.all(w[:, ] > 0):
                    Sigs[:, :, i] = Sigs_plane
                    break

    elif cov_type == 'diagonal':
        for i in range(NumberOfComponents):
            Sigs[:, :, i] = np.diag(np.random.random(d))

    # Solve for p(z | x, Theta(t))
    for k in range(NumberOfComponents):
        mvn = stats.multivariate_normal(Means[k, :], Sigs[:, :, k])
        pZ_X[:, k] = mvn.pdf(X) * Ps[k]

    pZ_X = pZ_X / pZ_X.sum(axis=1)[:, np.newaxis]  # np.newaxis fixes cannot broadcast (N,d) (N,) errors

    Diff = np.inf
    NumberIterations = 1

    # test effect of different covariance type
    # while Diff > DiffThresh:
    while Diff > DiffThresh and NumberIterations <= MaximumNumberOfIterations:

        # Update Means, Sigs, Ps
        MeansOld = Means.copy()
        SigsOld = Sigs.copy()
        PsOld = Ps.copy()
        for k in range(NumberOfComponents):
            # Means
            Means[k, :] = np.sum(X * pZ_X[:, k, np.newaxis], axis=0) / pZ_X[:, k].sum()

            # Sigs
            xDiff = X - Means[k, :]
            J = np.zeros((d, d))
            for i in range(N):
                J = J + pZ_X[i, k] * np.outer(xDiff[i, :], xDiff[i, :])
            Sigs[:, :, k] = J / pZ_X[:, k].sum()

            # Ps
            Ps[k] = pZ_X[:, k].sum() / N

        # Solve for p(z | x, Theta(t))
        for k in range(NumberOfComponents):
            mvn = stats.multivariate_normal(Means[k, :], Sigs[:, :, k])
            pZ_X[:, k] = mvn.pdf(X) * Ps[k]
        pZ_X = pZ_X / pZ_X.sum(axis=1)[:, np.newaxis]

        Diff = abs(MeansOld - Means).sum() + abs(SigsOld - Sigs).sum() + abs(PsOld - Ps).sum()

        NumberIterations = NumberIterations + 1

    # classify -- train_data
    distance_seotsa = np.zeros([train_data_seotsa.shape[0], Means.shape[0]])
    distance_versicolor = np.zeros([train_data_versicolor.shape[0], Means.shape[0]])
    distance_virginica = np.zeros([train_data_virginica.shape[0], Means.shape[0]])

    for i in range(Means.shape[0]):
        xx = train_data_seotsa[:, 0:4] - Means[i, :]
        yy = train_data_versicolor[:, 0:4] - Means[i, :]
        zz = train_data_virginica[:, 0:4] - Means[i, :]

        # fix bug - AttributeError: 'float' object has no attribute 'sqrt'
        xx = xx.astype('float64')
        yy = yy.astype('float64')
        zz = zz.astype('float64')

        # compute Euclidean Distance
        Euc_matrix_xx = np.linalg.norm(xx, ord=2, axis=1)
        Euc_matrix_yy = np.linalg.norm(yy, ord=2, axis=1)
        Euc_matrix_zz = np.linalg.norm(zz, ord=2, axis=1)

        distance_seotsa[:, i] = Euc_matrix_xx
        distance_versicolor[:, i] = Euc_matrix_yy
        distance_virginica[:, i] = Euc_matrix_zz

    classify_seotsa = np.sum(distance_seotsa, axis=0)
    classify_versicolor = np.sum(distance_versicolor, axis=0)
    classify_virginica = np.sum(distance_virginica, axis=0)

    # which column is the minimal one
    col_seotsa = classify_seotsa.argmin()
    col_versicolor = classify_versicolor.argmin()
    col_virginica = classify_virginica.argmin()

    # test part
    distance_test = np.zeros([test_data.shape[0], Means.shape[0]])
    true_value = np.array(test_data[:, 4]).reshape(test_data.shape[0], -1)
    predict_value = np.array(true_value)
    for i in range(Means.shape[0]):
        error = test_data[:, 0:4] - Means[i, :]

        # fix bug - AttributeError: 'float' object has no attribute 'sqrt'
        error = error.astype('float64')

        # compute Euclidean Distance
        Euc_matrix_test = np.linalg.norm(error, ord=2, axis=1)

        distance_test[:, i] = Euc_matrix_test
    # predict the class
    for j in range(test_data.shape[0]):
        mini_test_index = distance_test[j, :].argmin()
        if mini_test_index == col_seotsa:
            predict_value[j, 0] = 'Iris-setosa'
        elif mini_test_index == col_versicolor:
            predict_value[j, 0] = 'Iris-versicolor'
        elif mini_test_index == col_virginica:
            predict_value[j, 0] = 'Iris-virginica'
    result = np.hstack((true_value, predict_value))
    # digital result
    result_num = np.zeros((result.shape[0], 1))
    for i in range(result.shape[0]):
        if result[i, 0] == 'Iris-setosa':
            if result[i, 1] == 'Iris-setosa':
                result_num[i, :] = 1
            if result[i, 1] == 'Iris-versicolor':
                result_num[i, :] = -2
            if result[i, 1] == 'Iris-virginica':
                result_num[i, :] = -3
        elif result[i, 0] == 'Iris-versicolor':
            if result[i, 1] == 'Iris-setosa':
                result_num[i, :] = -10
            if result[i, 1] == 'Iris-versicolor':
                result_num[i, :] = 20
            if result[i, 1] == 'Iris-virginica':
                result_num[i, :] = -30
        elif result[i, 0] == 'Iris-virginica':
            if result[i, 1] == 'Iris-setosa':
                result_num[i, :] = -100
            if result[i, 1] == 'Iris-versicolor':
                result_num[i, :] = -200
            if result[i, 1] == 'Iris-virginica':
                result_num[i, :] = 300

    NumberIterations = NumberIterations - 1

    return Means, Sigs, Ps, pZ_X, result, result_num, Diff, NumberIterations


# K-fold cross-validation
def k_fold(data, n, k, NumberOfComponents, cov_type='isotropic', c=1):
    '''
    :param data: whole data
    :param n: how many data are using for cross-validation
    :param k: k fold
    :param NumberOfComponents: Number Of Components
    :param cov_type: type of covariance
    :param c: coefficient of isotropic type covariance
    '''
    # data preprocessing
    data = np.array(data)
    len_x = data.shape[0]

    # random the sequence
    list_i = np.random.permutation(len_x)

    # get a new random order data, X
    X = np.array(data)
    for i, j in enumerate(list_i):
        X[i, :] = data[j, :]

    # spilt new data into two groups: test set, cross-validation set
    test_data = X[n:150, 0:5]
    X_new = X[0:n, 0:5]

    # initialize a result map for cross-validation
    result_map_cv = np.zeros([k, int(X_new.shape[0] / k)])

    # initialize parameter of subplot
    num = np.ceil(k / 4)
    num_unit = 0
    # arrange = int(100 * num + 4 * 10)
    confusion_matrix = np.zeros([NumberOfComponents, NumberOfComponents])
    fig1 = plt.figure(figsize=(16, 9))
    fig1.suptitle('confusion matrix on {}-fold cross-validation'.format(k), fontsize=25, fontweight='bold')
    labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    label = ['setosa', 'versicolor', 'virginica']
    rate_list = []

    # get each valid_set and train_set
    if X_new.shape[0] % k != 0:
        print('({}-{})/{} must be int type'.format(X.shape[0], n, k))
        return
    elif (X_new.shape[0] / k) < 0:
        print('({}-{})/{} must be positive'.format(X.shape[0], n, k))
        return
    try:
        for i in range(int(X_new.shape[0] / k)):

            num_unit += 1

            valid_data = X_new[0 + i * k: k + i * k, 0:5]
            train_data = np.delete(X_new, np.s_[(0 + i * k):(k + i * k)], axis=0)

            # fit_EM and predict value by cross-validation
            while True:
                try:
                    Means, Sigs, Ps, pZ_X, result_cv, result_num_cv, diff_cv, num_iter_cv = EM_GaussianMixture(
                        train_data, valid_data, NumberOfComponents, cov_type, c)

                    # test performance in different covariance type
                    # print('error:', diff_cv)
                    # print('iteration times:', num_iter_cv)

                    result_map_cv[:, i] = result_num_cv.reshape(k, )
                    break
                except np.linalg.LinAlgError as e:
                    print('hold on,please.')

            # collect the confusion matrix
            C1 = metrics.confusion_matrix(result_cv[:, 0], result_cv[:, 1], labels=labels)
            df1 = pd.DataFrame(C1, index=labels, columns=labels)

            # plot
            plt.subplot(int(num), 4, int(num_unit))
            plt.imshow(df1, cmap=plt.cm.Blues)
            ticks = np.arange(len(label))
            plt.xticks(ticks, label, fontsize=10)
            plt.yticks(ticks, label, fontsize=8, rotation=90)
            plt.colorbar()
            items = np.reshape([[[i, j] for j in range(len(label))] for i in range(len(label))], (C1.size, 2))
            for i, j in items:
                plt.text(j, i, format(C1[i, j]), fontsize=18, fontweight='bold')
            rate = (np.sum(C1[0, 0] + C1[1, 1] + C1[2, 2])) / result_cv.shape[0]
            rate_list.append(rate)
            plt.title('correct rate is {:.2%}'.format(rate), y=-0.2, fontsize=14, fontweight='bold')
    except ZeroDivisionError as e:
        print('no validation set')

    plt.tight_layout()
    plt.savefig('./fig1.png', dpi=600)
    plt.show()

    print('total correct rate is {:.2%}'.format(np.average(rate_list)))

    # fit_EM by the whole train_data and predict value by test_data
    while True:
        try:
            Means, Sigs, Ps, pZ_X, result, result_num, diff, num_iter = EM_GaussianMixture(
                X_new, test_data, NumberOfComponents, cov_type, c)

            result_map = result_num.reshape(test_data.shape[0], )
            break
        except np.linalg.LinAlgError as e:
            print('hold on,please.')

    # initialize parameter of plot
    fig2 = plt.figure(figsize=(16, 9))
    plt.suptitle('confusion matrix on {} test data'.format(n), fontsize=25, fontweight='bold')
    labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    label = ['setosa', 'versicolor', 'virginica']

    # collect the confusion matrix
    C2 = metrics.confusion_matrix(result[:, 0], result[:, 1], labels=labels)
    df2 = pd.DataFrame(C2, index=labels, columns=labels)

    # plot
    plt.imshow(df2, cmap=plt.cm.Blues)
    ticks = np.arange(len(label))
    plt.xticks(ticks, label, fontsize=20)
    plt.yticks(ticks, label, fontsize=20, rotation=90)
    plt.colorbar()
    items = np.reshape([[[i, j] for j in range(len(label))] for i in range(len(label))], (C2.size, 2))
    for i, j in items:
        plt.text(j, i, format(C2[i, j]), fontsize=25, fontweight='bold')

    rate = (np.sum(C2[0, 0] + C2[1, 1] + C2[2, 2])) / result.shape[0]
    plt.title('correct rate is {:.2%}'.format(rate), y=-0.1, fontsize=25, fontweight='bold')
    plt.savefig('./fig2.png', dpi=600)
    plt.show()

    return result_map_cv, result_map


if __name__ == '__main__':
    # load the data
    data = pd.read_csv('./iris.data', header=None)
    X1 = np.array(data)[0:150, :]

    # matrix map
    # (setosa，setosa)=1,(setosa,versicolor)=-2,(setosa,virginica)=-3
    # (versicolor,setosa)=-10,(versicolor,versicolor)=20,(versicolor,virginica)=-30
    # (virginica,setosa)=-100,(virginica,versicolor)=-200,(virginica,virginica)=300
    start = time.time()
    # result_map_cv, result_map = k_fold(X1, 120, 10, 3, cov_type='isotropic')
    # result_map_cv, result_map = k_fold(X1, 120, 10, 3, cov_type='full')
    result_map_cv, result_map = k_fold(X1, 120, 10, 3, cov_type='isotropic', c=0.1)
    time_dur = time.time() - start
    print('The code run {:.0f}s'.format(time_dur % 60))
