from collections import Counter
from PIL import Image
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


class EHD(object):
    '''
    threshold, make the type of each pixel classing efficiently. it equal to 50 would be perfect
    :return
    bins, (17,5) for an image, the last row is mean values, which are mean values of each type of pixels in image
    '''

    def __init__(self, data, threshold=50):
        self.data = data
        self.threshold = threshold

    # return blocks_im
    def get_blocks(self):
        # split image into 4*4
        # resize image partition
        row_partition = math.ceil(int(str(self.data.shape[0])[:2]) / 4) \
                        * int(str(1) + (str(self.data.shape[0])[2:])) * 4
        col_partition = math.ceil(int(str(self.data.shape[1])[:2]) / 4) \
                        * int(str(1) + (str(self.data.shape[1])[2:])) * 4
        # get length of each block
        self.row_block = int(row_partition / 4)
        self.col_block = int(col_partition / 4)

        # get each partition
        num_row = int(row_partition / self.row_block)
        num_col = int(col_partition / self.col_block)

        blocks_im = []
        for i in range(num_row):
            for j in range(num_col):
                block_im = self.data[(self.row_block * i):((i + 1) * self.row_block)
                , (self.col_block * j):((j + 1) * self.col_block)]
                # make sure each partition shape is (40, 40)
                if (block_im.shape != (self.row_block, self.col_block)):
                    block_im_temp = np.zeros((self.row_block, self.col_block))
                    block_im_temp[0:block_im.shape[0], 0:block_im.shape[1]] = block_im
                    block_im = block_im_temp
                blocks_im.append(block_im)

        return blocks_im

    # return bin
    def get_bin(self, block_im):
        # get number of pixel
        num_pixel_row = int(self.row_block / 2)
        num_pixel_col = int(self.col_block / 2)

        # initialize bin for each block
        bin = np.zeros((1, 5))

        # set operators
        # v is vertical edge operator
        v = np.array([[1, -1], [1, -1]])
        # h is horizontal edge operator
        h = np.array([[1, 1], [-1, -1]])
        # d45 is diagonal 45 edge operator
        d45 = np.array([[2 ** 0.5, 0], [0, -(2 ** 0.5)]])
        # d135 is diagonal 135 edge operator
        d135 = np.array([[0, (2 ** 0.5)], [-(2 ** 0.5), 0]])
        # iso is isotropic edge operator
        iso = np.array([[2, -2], [-2, 2]])
        # make a list of operators
        list_operators = [v, h, d45, d135, iso]

        for i in range(num_pixel_row):
            for j in range(num_pixel_col):
                temp = np.zeros((1, 5))
                bin_pixel = np.zeros((1, 5))
                # extracting 2*2 pixel
                pixel = block_im[2 * i: 2 * (i + 1), 2 * j: 2 * (j + 1)]

                # apply operators
                for m in range(5):
                    temp[:, m] = abs(np.sum(pixel * list_operators[m]))

                if np.max(temp) > self.threshold:
                    # get the feature of each pixel
                    bin_pixel[:, np.argmax(temp)] = 1
                    bin = bin + bin_pixel
        return bin

    # return bins
    def runEHD(self):
        # initialize bins for image
        bins = np.zeros((17, 5))

        blocks_im = self.get_blocks()
        for i in range(len(blocks_im)):
            bin = self.get_bin(blocks_im[i])
            bins[i, :] = bin
        # compute means of bins in column, add means to the last row

        temp = bins[0:16, :]
        means = np.round(np.mean(temp, axis=0))
        bins[16, :] = means
        return bins


class KNN(object):
    '''
    :parameter
    target, a point which we want to find k closed points around
    features, features set of extract features
    k, how many closed points we want to find
    :return
    knn_labels, guess label
    index_data, k closed points indices in data,
    knn_result, dict type {lebel:number of points}.
    '''

    def __init__(self, features, data, labels, test_data, true_data=-1, k_nn=5):
        self.test_data = test_data
        self.true_data = true_data
        self.features = features
        self.data = data
        self.labels = labels
        self.k = k_nn

    def sort_distance(self):
        # Euclidean Distance
        distances = np.zeros((self.features.shape[0], 1))
        for i in range(self.features.shape[0]):
            distance = np.linalg.norm(self.test_data - self.features[i, :, :]) ** 2
            distances[i, 0] = distance
        # sort from small to large
        self.index = np.argsort(distances, axis=0)

    def run_knn(self):
        self.sort_distance()
        class_count = {}
        label_count = []
        for i in range(self.k):
            # get ture labels of the k closed points
            ture_label = int(self.labels.T[self.index[i]])

            # count points in each classed labels by knn, {label:number}
            class_count[ture_label] = class_count.get(ture_label, 0) + 1

            # get the whole labels of k closed points
            label_count.append(ture_label)

        # get the number of points in k closed point which are in this label
        self.knn_labels = Counter(label_count)
        # get the closed guess label，(closed label, points number)
        self.knn_result = self.knn_labels.most_common(1)

    def get_result(self):
        self.run_knn()
        self.index_data = []
        for i in range(self.k):
            # k closed points indices in data
            self.index_data.append(int(self.index[i]))

        return self.knn_result[0][0], self.index_data, dict(self.knn_labels)

    def true_label(self):
        # get the true label of target value
        exist_in_data = 0
        for i in range(self.features.shape[0]):
            if (self.test_data == self.features[i, :, :]).all():
                self.true_label = self.labels.T[i]
                exist_in_data = 1
        if (exist_in_data == 0):
            print('this image does not exist in Images set')
        elif (exist_in_data == 1):
            return self.true_label

    def visualize(self, test_original):
        # get the original data of the k closed points
        data_k_origin = np.zeros((len(self.index_data), self.data.shape[1], self.data.shape[2]))
        plt.figure(figsize=(16, 9), dpi=100)
        # plot the target data as main figure
        img_target = Image.fromarray(test_original)
        ax_main = plt.subplot2grid((2, 7), (0, 0), colspan=2, rowspan=2)
        ax_main.imshow(img_target)
        plt.axis('off')
        # name variables dynamic
        names = {}
        for i in range(1, len(self.index_data) + 1):
            names['ax%s' % i] = 1

        # subplot k closed data
        j = 0
        for key in names.keys():
            if j <= 10:
                index = self.index_data[j]
                img = Image.fromarray(self.data[index, :, :])
                key = plt.subplot2grid((2, 7), ((0 + j // 5), 2 + j % 5), colspan=1, rowspan=1)
                key.imshow(img)
                plt.axis('off')
            j += 1

        plt.show()


def score_knn(test_data, features, Images, Labels, k_closed, true_data=-1):
    predict_list = []
    true_list = []
    count = 0
    for i in range(test_data.shape[0]):
        # choose one picture
        x_test = test_data[i, :, :]

        # use K-NN classifier
        knn = KNN(features, Images, Labels, x_test, true_data, k_nn=k_closed)
        predict_label, index, knn_result = knn.get_result()
        true_label = int(knn.true_label())
        if (predict_label == true_label):
            count += 1
        predict_list.append(predict_label)
        true_list.append(true_label)
    result = np.vstack((true_list, predict_list))

    try:
        score = count / len(predict_list)
    except ZeroDivisionError as e:
        score = 0

    return score, result


def fold_generate(n, k_folds, shuffe=False):
    '''
    K-folds cross validation iterator.
    provides train/validation indices to split data in cross-validation.
    split dataset k consecutive folds(without shuffling by default).
    each fold is then use a validation set onve while the k - 1 remaining fold form the training set.
    :param n: number of data
    :param k_folds: K-folds
    :param shuffe:
    :return:
    train_index, valid_index
    '''
    order = np.arange(n)
    if shuffe:
        order = np.random.permutation(order)

    # folds have sine (cross-validation data // k_folds)
    fold_sizes = (n // k_folds) * np.ones(k_folds, dtype=int)
    # the first n % k_folds folds have size (cross-validation data // k_folds + 1)
    fold_sizes[: n % k_folds] += 1
    current = 0
    for fold_size in fold_sizes:
        # move one step forward
        start, stop = current, current + fold_size

        train_index = list(np.concatenate((order[:start], order[stop:])))
        valid_index = list(order[start:stop])
        yield train_index, valid_index
        current = stop


def split_data(data, labels, ratio):
    '''
    :param data: the whole extract feature value
    :param labels: the labels of data
    :param ratio: how many data are using for cross-validation
    :return:
    data_cv, labels_cv, data_test, labels_test
    '''
    # data preprocessing
    data = np.array(data)
    len_x = data.shape[0]

    # get the length of each type symbol
    list_temp = labels.tolist()
    list_labels = list_temp[0]
    counter_labels = Counter(list_labels)
    len_each_type = counter_labels.most_common()

    # get the spilt index
    list_num_cv = []
    list_num_test = []
    for i in range(len(len_each_type)):
        num_cv = int(len_each_type[i][1] * ratio)
        num_test = int(len_each_type[i][1] - num_cv)
        list_num_cv.append(num_cv)
        list_num_test.append(num_test)

    # initalize receptor
    data_cv = np.zeros((sum(list_num_cv), data.shape[1], data.shape[2]))
    labels_cv = np.zeros((labels.shape[0], sum(list_num_cv)))
    data_test = np.zeros((sum(list_num_test), data.shape[1], data.shape[2]))
    labels_test = np.zeros((labels.shape[0], sum(list_num_test)))
    count1 = 0
    count2 = 0
    count3 = 0
    for j in range(len(list_num_cv)):
        data_cv_temp = data[count1: (count1 + list_num_cv[j]), :, :]
        labels_cv_temp = labels[0, count1: (count1 + list_num_cv[j])]
        data_test_temp = data[count1: (count1 + list_num_test[j]), :, :]
        labels_test_temp = labels[0, count1: (count1 + list_num_test[j])]

        # get the cross-validation data and test data
        data_cv[count2: (count2 + list_num_cv[j]), :, :] = data_cv_temp
        labels_cv[:, count2: (count2 + list_num_cv[j])] = labels_cv_temp
        data_test[count3: (count3 + list_num_test[j]), :, :] = data_test_temp
        labels_test[:, count3: (count3 + list_num_test[j])] = labels_test_temp

        count1 += len_each_type[i][1]
        count2 += list_num_cv[j]
        count3 += list_num_test[j]
    return data_cv, labels_cv, data_test, labels_test


if __name__ == '__main__':

    # set parameters
    threshold = 50
    k_folds = 5
    k_closed = 2  # k_folds equal 2 will be best
    idx = 130  # for visualize

    # get the whole data
    Images = np.load('Images.npy')
    Labels = np.load('Labels.npy')

    # # extract feature of whole data set
    # features = np.zeros((Images.shape[0], 17, 5))
    # for i in range(0, Images.shape[0]):
    #     img = Images[i, :, :]
    #     edh = EHD(img, threshold=threshold)
    #     bins = edh.runEHD()
    #     features[i, :, :] = bins
    # np.save('features', features)

    features = np.load('features.npy')
    # # data preprocessing
    # list_features = np.arange(250)
    # m = 4  # from 0 to 24
    # start = 10 * m
    # stop = 10 + m * 10
    # list_features[start:stop]
    # num = m * 10
    # for j in list_features[start:stop]:
    #     print('{}:'.format(num), features[num, 16, :])
    #     num = num + 1

    # optimize knn by score curve
    # # plot - the whole data
    # test_data = features
    # score_list1 = []
    # for k in range(1, 11):
    #     score1, result1 = score_knn(test_data, features, Images, Labels, k)
    #     score_list1.append(score1)
    #
    # plt.figure(figsize=(20, 8), dpi=100)
    # plt.plot(range(1, 11), score_list1)
    # plt.xlabel('k')
    # plt.ylabel('score- the whole data')
    # plt.show()

    # plot - cross-validation data

    # # get the test data and cross-validation data
    # data_cv, labels_cv, data_test, labels_test = split_data(features, Labels, 0.8)
    # score_train = []
    # score_valid = []
    #
    # for k in range(1, 11):
    #     score_train_list = []
    #     score_valid_list = []
    #
    #     # get the index of train data index and validation data index
    #     kf = fold_generate(data_cv.shape[0], k_folds=k_folds)
    #     for train_index, valid_index in kf:
    #
    #         # get the X_train, X_valid, y_train, y_valid
    #         if k_folds == 1:
    #             X_train, X_valid = data_cv, data_cv
    #             y_train, y_valid = labels_cv, labels_cv
    #         else:
    #             X_train, X_valid = data_cv[train_index, :, :], data_cv[valid_index, :, :]
    #             y_train, y_valid = labels_cv[:, train_index], labels_cv[:, valid_index]
    #
    #         # get the score of X_train
    #         score2, result2 = score_knn(X_train, features, Images, Labels, k_closed=k)
    #         score_train_list.append(score2)
    #
    #         # get the score of X_valid
    #         score3, result3 = score_knn(X_valid, features, Images, Labels, k_closed=k)
    #         score_valid_list.append(score3)
    #
    #     # compute mean in each cross-validation
    #     score_train.append(np.mean(score_train_list))
    #     score_valid.append(np.mean(score_valid_list))

    # plot train data
    # plt.figure(figsize=(20, 8), dpi=100)
    # for i in range(0, 10):
    #     plt.plot(range(1, 11), score_train)
    #     plt.xlabel('k')
    #     plt.ylabel('score-tarin data')
    # plt.show()
    #
    # # plot validation data
    # plt.figure(figsize=(20, 8), dpi=100)
    # for i in range(0, 10):
    #     plt.plot(range(1, 11), score_valid)
    #     plt.xlabel('k')
    #     plt.ylabel('score-validation data')
    # plt.show()

    # # plot test data
    # score_list4 = []
    # for k in range(1, 11):
    #     score4, result4 = score_knn(data_test, features, Images, Labels, k)
    #     score_list4.append(score4)
    #
    # plt.figure(figsize=(20, 8), dpi=100)
    # plt.plot(range(1, 11), score_list4)
    # plt.xlabel('k')
    # plt.ylabel('score- the test data')
    # plt.show()

    # evaluate results using confusion matrices
    test_data = features
    score, result = score_knn(test_data, features, Images, Labels, k_closed=k_closed)
    confusion_matrix = np.zeros([result.shape[1], result.shape[1]])
    fig1 = plt.figure(figsize=(16, 9), dpi=100)
    fig1.suptitle('confusion matrix on {}-Nearest Neighbor'.format(k_closed), fontsize=25, fontweight='bold')
    labels = np.unique(result[0, :])

    # collect the confusion matrix
    C1 = metrics.confusion_matrix(result[0, :], result[1, :], labels=labels)
    df1 = pd.DataFrame(C1, index=labels, columns=labels)

    # plot
    plt.imshow(df1, cmap=plt.cm.Blues)
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, fontsize=10)
    plt.yticks(ticks, labels, fontsize=10)
    plt.colorbar()
    items = np.reshape([[[i, j] for j in range(len(labels))] for i in range(len(labels))], (C1.size, 2))
    sum = 0
    for i, j in items:
        if C1[i, j] != 0:
            plt.text(j, i, format(C1[i, j]), fontsize=10, fontweight='bold')
            if i == j:
                sum += C1[j, j]
    rate = sum / result.shape[1]
    plt.title('correct rate is {:.2%}'.format(rate), y=-0.1, fontsize=25, fontweight='bold')
    plt.show()

    # visualize

    # # extract feature
    # test_data = Images[idx, :, :]
    # test_ehd = EHD(test_data, threshold=threshold)
    # test_feature = test_ehd.runEHD()
    #
    # test_knn = KNN(features, Images, Labels, test_feature, k_nn=k_closed)
    # test_knn.get_result()
    # test_knn.visualize(test_data)
