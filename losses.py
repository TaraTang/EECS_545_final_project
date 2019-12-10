
#####################################
# 
#####################################

import numpy as np

def hamming_loss(true_labels, pred_labels):
    N, L = true_labels.shape
    return sum(sum(true_labels == pred_labels))/(N*L)

def accuracy(true_labels, pred_labels):
    N, L = true_labels.shape
    single_label = sum((true_labels == pred_labels).T)
    mask = np.ones_like(single_label)
    mask[single_label < L] = 0
    # print(mask)
    return sum(mask)/N

def evaluate_matrix(true_labels, pred_labels):
    N, L = true_labels.shape
    same_matrix = true_labels * pred_labels
    presicion = sum(sum(same_matrix))/sum(sum(pred_labels))
    recall = sum(sum(same_matrix))/sum(sum(true_labels))
    F1 = 2*presicion*recall/(presicion + recall)
    return presicion, recall, F1




