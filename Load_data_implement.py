#####################################
# 
#####################################

from scipy.io import arff
import pandas as pd
import numpy as np

##genbase data
features_genbase_train = np.load("features_genbase_train.npy")
features_genbase_test = np.load("features_genbase_test.npy")
labels_genbase_test = np.load("labels_genbase_test.npy")
labels_genbase_train = np.load("labels_genbase_train.npy")
##CAL500 data
features_CAL500_train = np.load("features_CAL500_train.npy")
features_CAL500_test = np.load("features_CAL500_test.npy")
labels_CAL500_test = np.load("labels_CAL500_test.npy")
labels_CAL500_train = np.load("labels_CAL500_train.npy")
##Corel5k data
features_Corel5k_train = np.load("features_Corel5k_train.npy")
features_Corel5k_test = np.load("features_Corel5k_test.npy")
labels_Corel5k_test = np.load("labels_Corel5k_test.npy")
labels_Corel5k_train = np.load("labels_Corel5k_train.npy")
##emotions data
features_emotions_train = np.load("features_emotions_train.npy")
features_emotions_test = np.load("features_emotions_test.npy")
labels_emotions_test = np.load("labels_emotions_test.npy")
labels_emotions_train = np.load("labels_emotions_train.npy")
##flags data
features_flags_train = np.load("features_flags_train.npy")
features_flags_test = np.load("features_flags_test.npy")
labels_flags_test = np.load("labels_flags_test.npy")
labels_flags_train = np.load("labels_flags_train.npy")
##scene data
features_scene_train = np.load("features_scene_train.npy")
features_scene_test = np.load("features_scene_test.npy")
labels_scene_test = np.load("labels_scene_test.npy")
labels_scene_train = np.load("labels_scene_train.npy")
##yeast data
features_yeast_train = np.load("features_yeast_train.npy")
features_yeast_test = np.load("features_yeast_test.npy")
labels_yeast_test = np.load("labels_yeast_test.npy")
labels_yeast_train = np.load("labels_yeast_train.npy")

##birds data
features_bird_train = np.load("features_bird_train.npy")
features_bird_test = np.load("features_bird_test.npy")
labels_bird_test = np.load("labels_bird_test.npy")
labels_bird_train = np.load("labels_bird_train.npy")

print('binary_relevance: ')
pred_labels = binary_relevance(features_bird_train, labels_bird_train, features_bird_test)
print("hamming loss: ", hamming_loss(labels_bird_test, pred_labels))
print("accuracy: ", accuracy(labels_bird_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_bird_test, pred_labels))

print('cc_logistic: ')
pred_labels = class_chain_log(features_bird_train, labels_bird_train, features_bird_test)
print("hamming loss: ", hamming_loss(labels_bird_test, pred_labels))
print("accuracy: ", accuracy(labels_bird_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_bird_test, pred_labels))

print('cc_svm: ')
pred_labels = class_chain_svm(features_bird_train, labels_bird_train, features_bird_test)
print("hamming loss: ", hamming_loss(labels_bird_test, pred_labels))
print("accuracy: ", accuracy(labels_bird_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_bird_test, pred_labels))

print('mcc_logistic: ')
pred_labels = modified_chain_log(features_bird_train, labels_bird_train, features_bird_test)
print("hamming loss: ", hamming_loss(labels_bird_test, pred_labels))
print("accuracy: ", accuracy(labels_bird_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_bird_test, pred_labels))

print('mcc_svm: ')
pred_labels = modified_chain_svm(features_bird_train, labels_bird_train, features_bird_test)
print("hamming loss: ", hamming_loss(labels_bird_test, pred_labels))
print("accuracy: ", accuracy(labels_bird_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_bird_test, pred_labels))

print('ecc: ')
ensemble = EnsembleClassifierChain(RandomForestClassifier())
ensemble.fit(features_bird_train, labels_bird_train)
pred_labels = ensemble.predict(features_bird_test)
print("hamming loss: ", hamming_loss(labels_bird_test, pred_labels))
print("accuracy: ", accuracy(labels_bird_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_bird_test, pred_labels))

##CAL500 data
features_CAL500_train = np.load("features_CAL500_train.npy")
features_CAL500_test = np.load("features_CAL500_test.npy")
labels_CAL500_test = np.load("labels_CAL500_test.npy")
labels_CAL500_train = np.load("labels_CAL500_train.npy")

print('binary_relevance: ')
pred_labels = binary_relevance(features_CAL500_train, labels_CAL500_train, features_CAL500_test)
print("hamming loss: ", hamming_loss(labels_CAL500_test, pred_labels))
print("accuracy: ", accuracy(labels_CAL500_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_CAL500_test, pred_labels))

print('cc_logistic: ')
pred_labels = class_chain_log(features_CAL500_train, labels_CAL500_train, features_CAL500_test)
print("hamming loss: ", hamming_loss(labels_CAL500_test, pred_labels))
print("accuracy: ", accuracy(labels_CAL500_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_CAL500_test, pred_labels))

print('cc_svm: ')
pred_labels = class_chain_svm(features_CAL500_train, labels_CAL500_train, features_CAL500_test)
print("hamming loss: ", hamming_loss(labels_CAL500_test, pred_labels))
print("accuracy: ", accuracy(labels_CAL500_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_CAL500_test, pred_labels))

print('mcc_logistic: ')
pred_labels = modified_chain_log(features_CAL500_train, labels_CAL500_train, features_CAL500_test)
print("hamming loss: ", hamming_loss(labels_CAL500_test, pred_labels))
print("accuracy: ", accuracy(labels_CAL500_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_CAL500_test, pred_labels))

print('mcc_svm: ')
pred_labels = modified_chain_svm(features_CAL500_train, labels_CAL500_train, features_CAL500_test)
print("hamming loss: ", hamming_loss(labels_CAL500_test, pred_labels))
print("accuracy: ", accuracy(labels_CAL500_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_CAL500_test, pred_labels))

print('ecc: ')
ensemble = EnsembleClassifierChain(RandomForestClassifier())
ensemble.fit(features_CAL500_train, labels_CAL500_train)
pred_labels = ensemble.predict(features_CAL500_test)
print("hamming loss: ", hamming_loss(labels_CAL500_test, pred_labels))
print("accuracy: ", accuracy(labels_CAL500_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_CAL500_test, pred_labels))

##emotions data
features_emotions_train = np.load("features_emotions_train.npy")
features_emotions_test = np.load("features_emotions_test.npy")
labels_emotions_test = np.load("labels_emotions_test.npy")
labels_emotions_train = np.load("labels_emotions_train.npy")

print('binary_relevance: ')
pred_labels = binary_relevance(features_emotions_train, labels_emotions_train, features_emotions_test)
print("hamming loss: ", hamming_loss(labels_emotions_test, pred_labels))
print("accuracy: ", accuracy(labels_emotions_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_emotions_test, pred_labels))

print('cc_logistic: ')
pred_labels = class_chain_log(features_emotions_train, labels_emotions_train, features_emotions_test)
print("hamming loss: ", hamming_loss(labels_emotions_test, pred_labels))
print("accuracy: ", accuracy(labels_emotions_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_emotions_test, pred_labels))

print('cc_svm: ')
pred_labels = class_chain_svm(features_emotions_train, labels_emotions_train, features_emotions_test)
print("hamming loss: ", hamming_loss(labels_emotions_test, pred_labels))
print("accuracy: ", accuracy(labels_emotions_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_emotions_test, pred_labels))

print('mcc_logistic: ')
pred_labels = modified_chain_log(features_emotions_train, labels_emotions_train, features_emotions_test)
print("hamming loss: ", hamming_loss(labels_emotions_test, pred_labels))
print("accuracy: ", accuracy(labels_emotions_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_emotions_test, pred_labels))

print('mcc_svm: ')
pred_labels = modified_chain_svm(features_emotions_train, labels_emotions_train, features_emotions_test)
print("hamming loss: ", hamming_loss(labels_emotions_test, pred_labels))
print("accuracy: ", accuracy(labels_emotions_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_emotions_test, pred_labels))

print('ecc: ')
ensemble = EnsembleClassifierChain(RandomForestClassifier())
ensemble.fit(features_emotions_train, labels_emotions_train)
pred_labels = ensemble.predict(features_emotions_test)
print("hamming loss: ", hamming_loss(labels_emotions_test, pred_labels))
print("accuracy: ", accuracy(labels_emotions_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_emotions_test, pred_labels))

##scene data
features_scene_train = np.load("features_scene_train.npy")
features_scene_test = np.load("features_scene_test.npy")
labels_scene_test = np.load("labels_scene_test.npy")
labels_scene_train = np.load("labels_scene_train.npy")

print('binary_relevance: ')
pred_labels = binary_relevance(features_scene_train, labels_scene_train, features_scene_test)
print("hamming loss: ", hamming_loss(labels_scene_test, pred_labels))
print("accuracy: ", accuracy(labels_scene_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_scene_test, pred_labels))

print('cc_logistic: ')
pred_labels = class_chain_log(features_scene_train, labels_scene_train, features_scene_test)
print("hamming loss: ", hamming_loss(labels_scene_test, pred_labels))
print("accuracy: ", accuracy(labels_scene_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_scene_test, pred_labels))

print('cc_svm: ')
pred_labels = class_chain_svm(features_scene_train, labels_scene_train, features_scene_test)
print("hamming loss: ", hamming_loss(labels_scene_test, pred_labels))
print("accuracy: ", accuracy(labels_scene_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_scene_test, pred_labels))

print('mcc_logistic: ')
pred_labels = modified_chain_log(features_scene_train, labels_scene_train, features_scene_test)
print("hamming loss: ", hamming_loss(labels_scene_test, pred_labels))
print("accuracy: ", accuracy(labels_scene_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_scene_test, pred_labels))

print('mcc_svm: ')
pred_labels = modified_chain_svm(features_scene_train, labels_scene_train, features_scene_test)
print("hamming loss: ", hamming_loss(labels_scene_test, pred_labels))
print("accuracy: ", accuracy(labels_scene_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_scene_test, pred_labels))

print('ecc: ')
ensemble = EnsembleClassifierChain(RandomForestClassifier())
ensemble.fit(features_scene_train, labels_scene_train)
pred_labels = ensemble.predict(features_scene_test)
print("hamming loss: ", hamming_loss(labels_scene_test, pred_labels))
print("accuracy: ", accuracy(labels_scene_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_scene_test, pred_labels))



##yeast data
features_yeast_train = np.load("features_yeast_train.npy")
features_yeast_test = np.load("features_yeast_test.npy")
labels_yeast_test = np.load("labels_yeast_test.npy")
labels_yeast_train = np.load("labels_yeast_train.npy")

print('binary_relevance: ')
pred_labels = binary_relevance(features_yeast_train, labels_yeast_train, features_yeast_test)
print("hamming loss: ", hamming_loss(labels_yeast_test, pred_labels))
print("accuracy: ", accuracy(labels_yeast_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_yeast_test, pred_labels))

print('cc_logistic: ')
pred_labels = class_chain_log(features_yeast_train, labels_yeast_train, features_yeast_test)
print("hamming loss: ", hamming_loss(labels_yeast_test, pred_labels))
print("accuracy: ", accuracy(labels_yeast_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_yeast_test, pred_labels))

print('cc_svm: ')
pred_labels = class_chain_svm(features_yeast_train, labels_yeast_train, features_yeast_test)
print("hamming loss: ", hamming_loss(labels_yeast_test, pred_labels))
print("accuracy: ", accuracy(labels_yeast_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_yeast_test, pred_labels))

print('mcc_logistic: ')
pred_labels = modified_chain_log(features_yeast_train, labels_yeast_train, features_yeast_test)
print("hamming loss: ", hamming_loss(labels_yeast_test, pred_labels))
print("accuracy: ", accuracy(labels_yeast_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_yeast_test, pred_labels))

print('mcc_svm: ')
pred_labels = modified_chain_svm(features_yeast_train, labels_yeast_train, features_yeast_test)
print("hamming loss: ", hamming_loss(labels_yeast_test, pred_labels))
print("accuracy: ", accuracy(labels_yeast_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_yeast_test, pred_labels))

print('ecc: ')
ensemble = EnsembleClassifierChain(RandomForestClassifier())
ensemble.fit(features_yeast_train, labels_yeast_train)
pred_labels = ensemble.predict(features_yeast_test)
print("hamming loss: ", hamming_loss(labels_yeast_test, pred_labels))
print("accuracy: ", accuracy(labels_yeast_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_yeast_test, pred_labels))

##genbase data
features_genbase_train = np.load("features_genbase_train.npy")
features_genbase_test = np.load("features_genbase_test.npy")
labels_genbase_test = np.load("labels_genbase_test.npy")
labels_genbase_train = np.load("labels_genbase_train.npy")
labels_genbase_train = np.delete(labels_genbase_train, [23,24], axis=1)
labels_genbase_test = np.delete(labels_genbase_test, [23,24], axis=1)

print('cc_logistic: ')
pred_labels = class_chain_log(features_genbase_train, labels_genbase_train, features_genbase_test)
print("hamming loss: ", hamming_loss(labels_genbase_test, pred_labels))
print("accuracy: ", accuracy(labels_genbase_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_genbase_test, pred_labels))

print('cc_svm: ')
pred_labels = class_chain_svm(features_genbase_train, labels_genbase_train, features_genbase_test)
print("hamming loss: ", hamming_loss(labels_genbase_test, pred_labels))
print("accuracy: ", accuracy(labels_genbase_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_genbase_test, pred_labels))

print('mcc_logistic: ')
pred_labels = modified_chain_log(features_genbase_train, labels_genbase_train, features_genbase_test)
print("hamming loss: ", hamming_loss(labels_genbase_test, pred_labels))
print("accuracy: ", accuracy(labels_genbase_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_genbase_test, pred_labels))

print('mcc_svm: ')
pred_labels = modified_chain_svm(features_genbase_train, labels_genbase_train, features_genbase_test)
print("hamming loss: ", hamming_loss(labels_genbase_test, pred_labels))
print("accuracy: ", accuracy(labels_genbase_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_genbase_test, pred_labels))

print('ecc: ')
ensemble = EnsembleClassifierChain(RandomForestClassifier())
ensemble.fit(features_genbase_train, labels_genbase_train)
pred_labels = ensemble.predict(features_genbase_test)
print("hamming loss: ", hamming_loss(labels_genbase_test, pred_labels))
print("accuracy: ", accuracy(labels_genbase_test, pred_labels))
print("evaluate_matrix: ", evaluate_matrix(labels_genbase_test, pred_labels))





