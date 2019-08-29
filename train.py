import pandas as pd
import numpy as np
import time
# from sklearn import cross_validation
from sklearn import datasets, svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection
import pickle
from sklearn.externals import joblib
# # from nolearn.dbn import DBN
# import timeit
#
# import gzip
# import os
# from helperFunctions import show_some_digits, plot_confusion_matrix, plot_param_space_scores

import struct
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
# import six
# from six.moves.urllib import request
from scipy.ndimage import interpolation

# from PreprocessingFunctions import remove_constant_pixels, deskew, moments

parent = 'http://yann.lecun.com/exdb/mnist'
imageFileTrain = 'train-images-idx3-ubyte'
labelFileTrain = 'train-labels-idx1-ubyte'
imageFileTest = 't10k-images-idx3-ubyte'
labelFileTest = 't10k-labels-idx1-ubyte'

def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix

def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    img =  interpolation.affine_transform(image,affine,offset=offset)
    # To renormalize the image values between 0 and 1
    return (img - img.min()) / (img.max() - img.min())

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

trainImage = read_idx(imageFileTrain)
trainLabel = read_idx(labelFileTrain)
testImage = read_idx(imageFileTest)
testLabel = read_idx(labelFileTest)

totalPixels = trainImage.shape[1]*trainImage.shape[2]

trainImage = trainImage.reshape(trainImage.shape[0],totalPixels)
testImage = testImage.reshape(testImage.shape[0],totalPixels)
# trainLabel = trainLabel.reshape(60000,1)
# testLabel = testLabel.reshape(10000,1)

# testing a randomforest classifier
seed = 7
np.random.seed(seed)

# print trainImage[0]
trainImage = trainImage/255.0
testImage = testImage/255.0

def deskewAll(X):
    currents = []
    for i in range(len(X)):
        currents.append(deskew(X[i].reshape(28,28)).flatten())
    return np.array(currents)
print len (trainImage)
# print trainImage[0].reshape(28,28)
trainImageDeskewed = deskewAll(trainImage)

testImageDeskewed = deskewAll(testImage)

print "deskew done"

svm_clsf = svm.SVC(C = 4, gamma = 0.04, kernel = 'rbf')
svm_clsf.fit(trainImageDeskewed, trainLabel)
svm_clsf.predict(testImageDeskewed)

expected = testLabel
predicted = svm_clsf.predict(testImageDeskewed)

print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))


joblib.dump(svm_clsf, 'model.joblib')
joblib.dump(svm_clsf, 'modelJOB.dat')
pickle.dump(svm_clsf, open("model.pkl", 'wb'))
pickle.dump(svm_clsf, open("modelPKL.dat", 'wb'))

#The Code below is for the Remove Pixel Portion of Preprocessing
# mergedData = np.append(trainImage, trainLabel, axis = 1)
#
# Columns = []
# for i in range (0, 784):
#     Columns.append("Pixel " + str(i + 1))
#
# Columns.append("Labels")
# #Incorporate all of the data into the Pandas Dataframe
# MNIST = pd.DataFrame(mergedData, columns = Columns)
#
# #Type into Jupiter Notebook for better formatting
# subset_pixels = MNIST.iloc[:, 1:]
# print subset_pixels.describe()
#
# MNISTwithoutLabels = MNIST.drop(columns = ['Labels'])
#
# def print_full(x):
#     pd.set_option('display.max_rows', len(x))
#     print(x)
#     pd.reset_option('display.max_rows')
#
#
# # print_full(MNISTwithoutLabels["Pixel 784"])
#
# # count = 0
# # for i in range(1, 785):
# #     if MNISTwithoutLabels["Pixel " + str(i)].min() == 255:
# #         print "Pixel " + str(i)
# #         count = count + 1
# #
# # print count
#
# MNISTdataDroppedPixels, dropped_pixels = remove_constant_pixels(MNISTwithoutLabels)
# trainImage = MNISTdataDroppedPixels.iloc[ :, :].values
# trainLabel = MNIST.iloc[ : ,784].values
# # print trainLabel
# # print MNIST.head()
#
# print trainImage.shape
# trainLabel  = trainLabel.flatten()
# print trainLabel.shape


# from mpl_toolkits.axes_grid1 import AxesGrid
# grid = AxesGrid(plt.figure(figsize=(8,15)), 141,  # similar to subplot(141)
#                     nrows_ncols=(10, 2),
#                     axes_pad=0.05,
#                     label_mode="1",
#                     )
#
# examples = (4181, 0), (3,1), (56282, 2), (25829,3), (9741,4) , (26901,5), (50027,6), (17935,7) , (41495, 8), (14662, 9)
# for examplenum,num in examples:
#     im = grid[2*num].imshow(trainImage[examplenum].reshape(28,28))
#     im2 = grid[2*num+1].imshow(deskew(trainImage[examplenum].reshape(28,28)))
#
# plt.subplot(1, 2, 1)
# plt.imshow(trainImage[3].reshape(28,28))
#
# newim = deskew(trainImage[3].reshape(28,28))
# plt.subplot(1, 2, 2)
# plt.imshow(newim)
# plt.show()

# def deskewAll(X):
#     currents = []
#     for i in range(len(X)):
#         currents.append(deskew(X[i].reshape(28,28)).flatten())
#     return np.array(currents)
# print len (trainImage)
# # print trainImage[0].reshape(28,28)
# trainImageDeskewed = deskewAll(trainImage)
# testImageDeskewed = deskewAll(testImage)
#
# # print trainImage[0]
#
# gamma_range = np.outer(np.logspace(-3, 0, 4),np.array([1,5]))
# gamma_range = gamma_range.flatten()
# gamma_range = np.array([0.04])
#
# # generate matrix with all C
# #C_range = np.outer(np.logspace(-3, 3, 7),np.array([1,2, 5]))
# C_range = np.outer(np.logspace(-1, 1, 3),np.array([1,5]))
# # flatten matrix, change to 1D numpy array
# C_range = np.array([4])
#
# print C_range, gamma_range
#
# parameters = {'kernel':['rbf'], 'C':C_range, 'gamma': gamma_range}
# from sklearn.model_selection import KFold
# svm_clsf = svm.SVC()
# kf = KFold(n_splits=2, shuffle= True)
# grid_clsf = GridSearchCV(estimator=svm_clsf, param_grid=parameters, n_jobs=1, verbose=2, cv = kf)
#
#
# start_time = dt.datetime.now()
# print('Start param searching at {}'.format(str(start_time)))
#
# grid_clsf.fit(trainImageDeskewed, trainLabel)
#
# elapsed_time= dt.datetime.now() - start_time
# print('Elapsed time, param searching {}'.format(str(elapsed_time)))
# sorted(grid_clsf.cv_results_.keys())
#
# classifier = grid_clsf.best_estimator_
# params = grid_clsf.best_params_
#
# scores = grid_clsf.cv_results_['mean_test_score'].reshape(len(C_range),
#                                                      len(gamma_range))
#
# expected = testLabel
# predicted = classifier.predict(testImageDeskewed)
#
# show_some_digits(testImageDeskewed, predicted, title_text="Predicted {}")
#
# print("Classification report for classifier %s:\n%s\n"
#       % (classifier, metrics.classification_report(expected, predicted)))
#
# cm = metrics.confusion_matrix(expected, predicted)
# print("Confusion matrix:\n%s" % cm)
#
# plot_confusion_matrix(cm)
#
#
# pickle.dump(grid_clsf, open('rbf_deskewed_test', 'wb'))
# pickle.dump(grid_clsf.best_estimator_, open('rbf_best_deskewedtest', 'wb'))
#
# print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))
#
# print "the scores are ", scores
# print "the C range is  ", C_range
# print "the gamma range is  ", gamma_range
# print "the version is deskewTest"
# joblib.dump(grid_clsf.best_estimator_, 'grid_clsf_best_estimator_rbf_deskewedtest.pkl')
# joblib.dump(grid_clsf, 'grid_clsf_full_rbf_deskewedtest.pkl')
#
#
# plot_param_space_scores(scores, C_range, gamma_range)
