import numpy as np
import label_power_set as lpset
import random
import losses as los

#####################################
# written by Yicheng Tao
#####################################

def comput_u_theta(X_dn):
    X_dn = X_dn.T
    n = X_dn.shape[1]
    U_v = 1/n*np.sum(X_dn, axis=1).reshape(-1, 1)
    X_c = X_dn - U_v
    theta_2_v = (np.sum(X_c*X_c, axis=1)/(n-1)).reshape(-1, 1)
    recp_theta_2_v = 1/theta_2_v
    A = np.diag(np.sqrt(recp_theta_2_v.reshape(-1, )))
    X_sphere = np.dot(A, X_c)
    return U_v.T, theta_2_v.T, X_sphere.T

def label_space_partition(X_train, Y_train, X_test, k):
    D = Y_train.shape[1]
    print(D)
    rand_list = divide_list(D, k)
    print(rand_list)
    test_n = X_test.shape[0]
    test_rst = np.zeros((test_n, D))

    for i in range(len(rand_list)):
        label_t = Y_train[:, rand_list[i]]
        mod = lpset.model_train(X_train, label_t)
        rst_m = lpset.predict_model(X_test, mod, len(rand_list[i]))
        # print(rst_m.shape)
        for j in range(len(rand_list[i])):
            test_rst[:, rand_list[i][j]] = rst_m[:, j]
    return test_rst


def divide_list(listlen, k):
    A =list(range(listlen))
    A_out = []
    for i in range(int(listlen/k-1)):
        slice = random.sample(A, k)
        A_out.append(slice)
        for i in range(k):
            A.remove(slice[i])

    A_out.append(A)
    return A_out

#
label_all = np.load("labels_bird_train.npy")
x_data = np.load("features_bird_train.npy").astype("float64")
test = np.load("features_bird_test.npy").astype("float64")
label_true = np.load("labels_bird_test.npy")
print(x_data.shape)
print(test.shape)
U, theta, x_data_sp = comput_u_theta(x_data)
test_sp = (test-U)/np.sqrt(theta)

res = label_space_partition(x_data_sp, label_all, test_sp, 3) # label space partition
# res = label_space_partition(x_data_sp, label_all, test_sp, 10) # label power set
print("bird: ")
h_l = los.hamming_loss(label_true, res)
print(h_l)

acc = los.accuracy(label_true, res)
print(acc)

P,R, F = los.evaluate_matrix(label_true, res)
print(P, R, F)

