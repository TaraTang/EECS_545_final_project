import numpy as np
import losses as los
from tqdm import*

#####################################
# 
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

def compu_prior_prob(Y_l, s):
    P_H1_list = []
    P_H0_list = []
    l = Y_l.shape[1]
    m = Y_l.shape[0]
    for i in range(l):
        P_H1l = (s + np.sum(Y_l[:, i]))/ (2 * s + m)
        P_H0l = 1 - P_H1l
        P_H1_list.append(P_H1l)
        P_H0_list.append(P_H0l)

    return P_H1_list, P_H0_list


def compu_post_prob(X_t, Y_l, s, k):
    knn_list = KNN_train(X_t, k)
    lgth = Y_l.shape[1]
    m = Y_l.shape[0]
    P_EH_L_1 = []
    P_EH_L_0 = []
    KK = k
    for l in tqdm(range(lgth)):
        P_EH_J_1 = []
        P_EH_J_0 = []
        c = np.zeros((KK+1,))
        c_p = np.zeros((KK+1,))
        for i in range(m):
            N_xi_L = Y_l[knn_list[i], :]
            f = np.sum(N_xi_L[:, l])
            if Y_l[i, l] == 1:
                c[f] = c[f] + 1
            else:
                c_p[f] = c_p[f] + 1
            # for ki in range(N_xi_L.shape[0]):
            #     if N_xi_L[ki, l] == 1:
            #         c[f] = c[f] + 1
            #     else:
            #         c_p[f] = c_p[f] + 1
        for j in range(KK+1):
            EH_1 = (s + c[j]) / (s*(KK+1)+np.sum(c))
            EH_0 = (s + c_p[j]) / (s * (KK+1) + np.sum(c_p))
            P_EH_J_1.append(EH_1)
            P_EH_J_0.append(EH_0)
        P_EH_L_1.append(P_EH_J_1)
        P_EH_L_0.append(P_EH_J_0)
    return np.array(P_EH_L_1), np.array(P_EH_L_0)


def KNN_train(train_X, k):
    res = []
    for i in tqdm(range(train_X.shape[0])):
        x_i = train_X[i, :].reshape(1, -1) # R 1*d
        x_i_2 = np.dot(x_i, x_i.T) * np.ones((train_X.shape[0], 1)) # R n*1
        d = np.sum(train_X * train_X, axis=1).reshape(-1, 1) # R n *1
        D = x_i_2 - 2 * np.dot(train_X, x_i.T) + d
        idx = np.argsort(D.reshape(-1, ))[1:k+1]
        res.append(idx)

    return res

def KNN_train_test(train_X, k, test_X):
    res = []
    for i in range(test_X.shape[0]):
        x_i = test_X[i, :].reshape(1, -1)
        x_i_2 = np.dot(x_i, x_i.T) * np.ones((train_X.shape[0], 1))
        d = np.sum(train_X * train_X, axis=1).reshape(-1, 1)
        D = x_i_2 - 2 * np.dot(train_X, x_i.T) + d
        idx = np.argsort(D.reshape(-1, ))[0:k]
        res.append(idx)

    return res

def ml_KNN(train_X, Y_l, test_X, k, s):
    ph_1, ph_0 = compu_prior_prob(Y_l, s)
    # print(ph_1)
    # print(ph_0)
    print("prior done")
    P_eh_1, P_eh_0 = compu_post_prob(train_X, Y_l, s, k)
    # print(P_eh_1)
    # print(P_eh_0)
    print("post done")
    test_X_res = KNN_train_test(train_X, k, test_X)
    lgth = Y_l.shape[1]
    test_L = np.zeros((len(test_X_res), lgth))
    for t in tqdm(range(len(test_X_res))):
        for l in range(lgth):
            ctl = int(np.sum(Y_l[test_X_res[t], l]))
            y_l_1 = ph_1[l] * P_eh_1[l, ctl]
            y_l_0 = ph_0[l] * P_eh_0[l, ctl]
            # print("PEH: ", P_eh_1[l, ctl], P_eh_0[l, ctl])
            # print("PH: ", ph_1[l], ph_0[l])
            if y_l_1 > y_l_0:
                test_L[t, l] = 1
            else:
                test_L[t, l] = 0
    return test_L


t_X = np.load("features_genbase_train.npy").astype("float64")
tst_X = np.load("features_genbase_test.npy").astype("float64")
Y_lab = np.load("labels_genbase_train.npy")
tst_lab = np.load("labels_genbase_test.npy")
print(Y_lab.shape)

U, theta, x_data_sp = comput_u_theta(t_X)
test_sp = (tst_X-U)/np.sqrt(theta)
tst_L = (ml_KNN(x_data_sp, Y_lab, test_sp, 10, 1))
print("genbase: ")
h_l = los.hamming_loss(tst_lab, tst_L)
print(h_l)
acc = los.accuracy(tst_lab, tst_L)
print(acc)
P,R, F = los.evaluate_matrix(tst_lab, tst_L)
print(P, R, F)

