import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm

#####################################
# written by Yicheng Tao
#####################################


def model_train(X_f, labels):
    Y = transfer_labels(labels)
    # print(Y)
    # model = svm.SVC(C=15, kernel='rbf', gamma=0.03, tol=1e-10, decision_function_shape='ovo', max_iter=10000,
    #                 cache_size=10000)
    model = LogisticRegression()
    # model = svm.SVC()
    model.fit(X_f, Y)

    return model

def predict_model(X_f, model, D):
    rst = model.predict(X_f)
    # rst = np.array(rst).reshape(-1, 1)
    inv_rst = inverse_labels(rst, D)
    return inv_rst


def transfer_labels(label_m):
    (N, d) = label_m.shape
    label_tf = []
    for i in range(N):
        label_s = "0"
        for j in range(d):
            label_s = label_s + "{}".format(label_m[i, j])
        label_tf.append(label_s)
    label_tf = np.array(label_tf)
    return label_tf


def inverse_labels(label_trsf, D):

    N = label_trsf.shape[0]
    label_inv = np.zeros((N, D))
    for i in range(N):
        for j in range(D):
            label_inv[i, j] = int(label_trsf[i][j + 1])
    return label_inv

# label_all = np.load("labels_CAL500_train.npy")
# print(label_all.shape)
# label_tf = []
# for i in range(400):
#     label_s = "0"
#     for j in range(174):
#         label_s = label_s + "{}".format(label_all[i, j])
#     label_tf.append(label_s)
# label_tf = np.array(label_tf)
#
# print(label_tf[0][0])
#
# label_inv = np.zeros((400, 174))
# for i in range(400):
#     for j in range(174):
#         label_inv[i, j] = int(label_tf[i][j+1])
# print(np.sum(label_inv-label_all))

#
# label_transf = transfer_labels(label_all)
# print(label_transf)
# lab_inv = inverse_labels(label_transf, 174)
# print(np.sum(label_all-lab_inv))

# x_data = np.load("features_CAL500_train.npy").astype("float64")
# model = LogisticRegression()
# model.fit(x_data, label_tf)