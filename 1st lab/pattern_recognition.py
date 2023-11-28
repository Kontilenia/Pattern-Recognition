# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 19:44:30 2022

@author: User
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

from sklearn.decomposition import PCA  # to apply PCA
import seaborn as sns  # to plot the heat maps


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import BaggingClassifier

import scipy.stats as sts

# gridspec contains classes that help to layout multiple Axes
# in a grid-like pattern within a figure.
# The GridSpec specifies the overall grid structure.
# Individual cells within the grid are referenced by SubplotSpecs.


# Step 1

# Find data path combining file absolute path and data folder
absolute_path = os.path.dirname(__file__)
data_path = ["test.txt", "train.txt"]
data_folder = "data"
full_path_test = os.path.join(absolute_path, data_folder, data_path[0])
full_path_train = os.path.join(absolute_path, data_folder, data_path[1])

test = pd.read_csv(full_path_test, sep=' ', header=None, index_col=False)
test.dropna(axis=1, how='all', inplace=True)
y_test = test.iloc[:, 0]
X_test = test.iloc[:, 1:]

train = pd.read_csv(full_path_train, sep=' ', header=None, index_col=False)
train.dropna(axis=1, how='all', inplace=True)
y_train = train.iloc[:, 0]
X_train = train.iloc[:, 1:]

X_train1 = X_train.to_numpy()
X_test1 = X_test.to_numpy()

# Step 2
# Print digit
digit = X_train1[131, :]
digit = np.reshape(digit, (16, 16))  # Reshape digit from 1X256 TO 16X16
plt.imshow(digit, cmap='gray')
plt.show()


ten_list = []
for i in range(10):
    random_sample = y_train[y_train == i].sample(n=1)
    index = random_sample.index
    new = np.reshape(X_train1[index, :], (16, 16))
    ten_list.append(new)

fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(10, 8))
gs1 = gridspec.GridSpec(4, 4)
gs1.update(wspace=0.025, hspace=0.05)
for i, ax in enumerate(axs.ravel()):
    ax.imshow(ten_list[i], cmap='gray')
plt.subplots_adjust(wspace=0.01, hspace=0)
plt.show()

conv_index = (0 - 1) * 16 + 0
zero_pixel_mean = X_train1[y_train == 0][:, conv_index]
mean_val = np.mean(zero_pixel_mean, axis=0)

# Step 5 Standard deviation δεδομένου του 4

zero_pixel_sd = X_train1[y_train == 0][:, conv_index]
var = np.var(zero_pixel_sd, axis=0)

print("Standard deviation: ", var)
print("mean: ", mean_val)

# Step 6: mean for each pixel of all the same digits
digit1 = 0  # Η εκφώνηση λέει το ψηφίο 0
mean_of_digit1 = np.mean(X_train1[y_train == digit1], axis=0)

# Step 6: Standard deviation for each pixel of all the same digits
sd_of_digit1 = np.var(X_train1[y_train == digit1], axis=0)

# Step 7
schema_med = np.reshape(mean_of_digit1, (16, 16))
print("Zero digit using the mean values: ")
plt.imshow(schema_med, cmap='gray')
plt.show()

# Step 8
schema_sd = np.reshape(sd_of_digit1, (16, 16))
print("Zero digit using the standard deviation values: ")
plt.imshow(schema_sd, cmap='gray')
plt.show()

# Step 9 (a)
mean_values = []
var_values = []
for i in range(0, 10):
    mean_variable = np.mean(X_train1[y_train == i], axis=0)
    mean_values.append(mean_variable)
    var_variable = np.var(X_train1[y_train == i], axis=0)
    var_values.append(var_variable)

# Step 9 (b)
fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(10, 10))
gs1 = gridspec.GridSpec(4, 4)
gs1.update(wspace=0.025, hspace=0.05)
for i, ax in enumerate(axs.ravel()):
    ax.imshow(mean_values[i].reshape(16, 16), cmap='gray')
    plt.subplots_adjust(wspace=0.01, hspace=0)
plt.show()

# Step 10

euclidian_distance = []
mean_values_np = np.array(mean_values)
for i in range(10):
    # Euclidian distance between mean and the element 101
    euclidian_distance.append(np.linalg.norm(mean_values_np[i] - X_test1[101, :]))

# Select class base on the minimum distance
print("Predicted class is: " + str(euclidian_distance.index(min(euclidian_distance))))
print("Real class class is: " + str(y_test.iloc[101]))

# Step 11 (a)

all_euclidian_distance = []
# store the final prediction for each digit of x_test to compare it with 0-9 digits
for j in range(len(y_test)):
    # is used for each specific digit of x_test in order
    specific_euclidian_distance = np.linalg.norm(mean_values_np - X_test1[j, :], axis=1)
    all_euclidian_distance.append(np.argmin(specific_euclidian_distance))

# Step 11 (b)
print("Success Performance of Euclidian Distance method is: {:.2%}".format(
    len(y_test[y_test == all_euclidian_distance]) / len(y_test)))


# Step 12

class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self):
        self.X_mean_ = None

    def fit(self, X, y):

        mean_value = []
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        for i in range(0, 10):
            mean_variables = np.mean(X[y == i], axis=0)
            mean_value.append(mean_variables)
        self.X_mean_ = np.array(mean_value)
        return self

    def predict(self, X):
        
        predictions = []
        # store the final prediction for each digit of x_test to compare it with 0-9 digits
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        for i in range(X.shape[0]):
            # is used for each specific digit of x_test in order
            euclidian_distances = np.linalg.norm(self.X_mean_ - X[i, :], axis=1)
            predictions.append(np.argmin(euclidian_distances))
        return predictions

    def score(self, X, y, sample_weight=None):
        
        pred = self.predict(X)
        return len(y[y == pred]) / len(y)


simple_model = EuclideanDistanceClassifier()
model = simple_model.fit(X_train, y_train)
score = simple_model.score(X_test, y_test)
print("Success Performance of Euclidian Distance method is: {:.2%}".format(score))

# Step 13 (a)

cv_model = EuclideanDistanceClassifier()
scores = cross_val_score(cv_model, X_train, y_train, cv=5)
print("Success Performance of Euclidian Distance method with 5-CV is: {:.2%}".format(np.mean(scores)))
# Step 13 (b)

pca = PCA(n_components=2)
pca.fit(X_train)
X_train_curtailed = pca.transform(X_train)
pca.fit(X_test)
X_test_curtailed = pca.transform(X_test)

# Checking Co-relation between features after PCA
Train_curtailed = pd.DataFrame(X_train_curtailed, columns=['Feature 1', 'Feature 2'])
new = sns.heatmap(Train_curtailed.corr(), cmap="Blues")
new.xaxis.tick_top()
plt.show()


def plot_clf(clf, X, y):
    fig, ax = plt.subplots()
    # title for the plots
    title = 'Decision surface'
    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]

    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                         np.arange(y_min, y_max, .05))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z)  # to work with lists, because our class returns lists
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    for i in range(10):
        ax.scatter(X0[y == i], X1[y == i], label='Digit ' + str(i), s=18, alpha=1.0, edgecolors='k')

    ax.set_ylabel('Random feature 1')
    ax.set_xlabel('Random feature 2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend(loc="best")
    plt.show()


pca_model = EuclideanDistanceClassifier()
pca_model = pca_model.fit(X_train_curtailed, y_train)
plot_clf(pca_model, X_train_curtailed, y_train)

# Step 13 (c)
train_sizes, train_scores, test_scores = learning_curve(cv_model, X_train, y_train, cv=5, n_jobs=-1,
                                                        train_sizes=np.linspace(.1, 1.0, 5))


def plot_learning_curve(train_score, test_score, train_size, ylim=(0, 1)):
    plt.figure()
    plt.title("Learning Curve")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_score, axis=1)
    train_scores_var = np.var(train_score, axis=1)
    test_scores_mean = np.mean(test_score, axis=1)
    test_scores_var = np.var(test_score, axis=1)
    plt.grid()

    plt.fill_between(train_size, train_scores_mean - train_scores_var,
                     train_scores_mean + train_scores_var, alpha=0.1,
                     color="r")
    plt.fill_between(train_size, test_scores_mean - test_scores_var,
                     test_scores_mean + test_scores_var, alpha=0.1, color="g")
    plt.plot(train_size, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_size, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()  


plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(.7, .9))
print()

# Step 14
all_apriori = []
[all_apriori.append(len(y_train[y_train == i]) / len(y_train)) for i in range(0, 10)]


# Step 15
def calculate_probability(x, mean, var):

    return sts.norm.pdf(x, mean, np.sqrt(var))
   

class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.X_mean_ = None
        self.X_var_ = None
        self.X_apriori_ = None
        self._var_smoothing_ = 1e-5

    def fit(self, X, y, nb_var=None):

        mean_value = []
        var_value = []
        apriori_value = []
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
            
        for i in range(0, 10):
            mean_variables = np.mean(X[y == i], axis=0)
            mean_value.append(mean_variables)
            if nb_var is None:
                var_variables = np.var(X[y == i], axis=0)
                var_value.append(var_variables)
            else:
                var_value.append(np.ones(256) * nb_var)
            apriori_value.append(len(y[y == i]) / len(y))

        var_value = np.array(var_value)

        small_number = 0.0
        if nb_var is None:
            small_number = self._var_smoothing_ * var_value.max()

        var_value += small_number

        self.X_mean_ = np.array(mean_value)
        self.X_var_ = var_value
        self.X_apriori_ = np.array(apriori_value)
        return self

    def predict(self, X):
        
        probabilities = []
        predictions = []  
        # store the final prediction for each digit of x_test
        # to compare it with 0-9 digits
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        for i in range(X.shape[0]):
            for k in range(0, 10):
                a = 1
                for j in range(X.shape[1]):
                    a = a * calculate_probability(X[i, j], self.X_mean_[k, j], self.X_var_[k, j])
                probabilities.append(a * self.X_apriori_[k])
            predictions.append(np.argmax(probabilities))
            probabilities.clear()
        return predictions

    def score(self, X, y, sample_weight=None):
        
        pred = self.predict(X)
        return len(y[y == pred]) / len(y)


# Naive Bayes
Bayes = NaiveBayesClassifier()
Bayes = Bayes.fit(X_train, y_train)
Bayes_score = Bayes.score(X_test, y_test)
print("Success Performance of Naive Bayes method is: {:.2%}".format(Bayes_score))

# Naive Bayes for variance equals 1
Bayes2 = NaiveBayesClassifier()
Bayes2 = Bayes2.fit(X_train, y_train, nb_var=1)
Bayes_score2 = Bayes2.score(X_test, y_test)
print("Success Performance of Naive Bayes method (for variance equals 1) is: {:.2%}".format(Bayes_score2))

classifiers = {}
# Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
classifiers['Nayve Bayes'] = gnb.score(X_test, y_test)
# KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)  # default value
knn.fit(X_train, y_train)
classifiers['KNeighborsClassifier'] = knn.score(X_test, y_test)
# SVM linear kernel
svc_model_linear = SVC(C=1.0, random_state=1, kernel='linear', probability=True)
svc_model_linear.fit(X_train, y_train)
classifiers['SVM linear kernel'] = svc_model_linear.score(X_test, y_test)
# SVM RBF kernel
svm_rbf = SVC(kernel='rbf', random_state=1, gamma=0.02, C=1, probability=True)
svm_rbf.fit(X_train, y_train)
classifiers['SVM RBF kernel'] = svm_rbf.score(X_test, y_test)
# SVM sigmoid kernel
svm_sigmoid = SVC(kernel='sigmoid')  # ,probability=True)
svm_sigmoid.fit(X_train, y_train)
classifiers['SVM sigmoid kernel'] = svm_sigmoid.score(X_test, y_test)

sorted_accuracy = [(k, classifiers[k]) for k in sorted(classifiers, key=classifiers.get, reverse=True)]
print("Success of each classifier follows below in sorted list:")
for k, v in sorted_accuracy:
    print("Model: {} has performance: {:.2%}".format(k, v))


# Here we try to plot a figure with all confusion matricrs of 6 classifiers to compare
# and decide which we will choose to our VotinClassifier

def plot_confusion_matrices(X_tr, y_tr, X_te, y_te):
    classifiers = {1: SVC(kernel="linear"),
                   2: SVC(kernel="rbf"),
                   3: SVC(kernel="sigmoid"),
                   4: DecisionTreeClassifier(),
                   5: KNeighborsClassifier(n_neighbors=5),
                   6: GaussianNB()}
    class_labels = {1: 'Linear SVM',
                    2: 'RBF SVM',
                    3: 'Sigmoid SVM',
                    4: 'Decision Tree',
                    5: 'kNN (k = 5)',
                    6: 'Naive Bayes (sklearn)'}
    for key in classifiers:
        classifiers[key].fit(X_tr, y_tr)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
    digits = np.arange(10)

    for key, ax in zip(classifiers, axes.flatten()):
        plot_confusion_matrix(classifiers[key], X_te, y_te, ax=ax, cmap='viridis', display_labels=digits)
        ax.title.set_text(class_labels[key])
    plt.tight_layout()
    plt.show()
    return


plot_confusion_matrices(X_train, y_train, X_test, y_test)

estimator = [('SVM RBF kernel', svm_rbf), ('SVM linear kernel', svc_model_linear), ('KNeighborsClassifier', knn)]

vot_soft = VotingClassifier(estimators=estimator, voting='soft')
vot_soft.fit(X_train, y_train)
y_pred1 = vot_soft.predict(X_test)
score1 = accuracy_score(y_test, y_pred1)
print("Soft Voting Score {:.2%} ".format(score1))

vot_hard = VotingClassifier(estimators=estimator, voting='hard')
vot_hard.fit(X_train, y_train)
y_pred2 = vot_hard.predict(X_test)
score2 = accuracy_score(y_test, y_pred2)
print("Hard Voting Score {:.2%} ".format(score2))

estimator2 = [('Gaussian Naive', gnb), ('SVM sigmoid kernel', SVC(kernel='sigmoid', random_state=1, probability=True)),
              ('KNeighborsClassifier', knn)]

vot_soft2 = VotingClassifier(estimators=estimator2, voting='soft')
vot_soft2.fit(X_train, y_train)
y_pred3 = vot_soft2.predict(X_test)
score3 = accuracy_score(y_test, y_pred3)
print("Soft Voting Score {:.2%} ".format(score3))

vot_hard2 = VotingClassifier(estimators=estimator2, voting='hard')
vot_hard2.fit(X_train, y_train)
y_pred4 = vot_hard2.predict(X_test)
score4 = accuracy_score(y_test, y_pred4)
print("Hard Voting Score {:.2%} ".format(score4))

# Bootstrap Aggregation
estimator_range = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 80, 100]
models = []
scoring = []
for n_estimators in estimator_range:
    # Create bagging classifier
    clf = BaggingClassifier(n_estimators=n_estimators, random_state=22)
    # Fit the model
    clf.fit(X_train, y_train)
    # Append the model and score to their respective list
    models.append(clf)
    scoring.append(accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))
# Generate the plot of scores against number of estimators
plt.figure(figsize=(9, 6))
plt.plot(estimator_range, scoring)
plt.xlabel("n_estimators", fontsize=18)
plt.ylabel("score", fontsize=18)
plt.tick_params(labelsize=16)
plt.show()

bagging = {}
# Naive Bayes
bagging_gnb = BaggingClassifier(base_estimator =gnb,n_estimators =24,random_state = 22)
bagging['Nayve Bayes'] = cross_val_score(bagging_gnb, X_train, y_train, cv = 5).mean()
# KNeighborsClassifier
bagging_knn = BaggingClassifier(base_estimator = knn,n_estimators =24,random_state = 22) #default value
bagging['KNeighborsClassifier'] = cross_val_score(bagging_knn, X_train, y_train, cv = 5).mean()
# SVM linear kernel
bagging_linear = BaggingClassifier(base_estimator = SVC(C=1.0, random_state=1, kernel='linear'),n_estimators =24,random_state = 22)
bagging['SVM linear kernel'] = cross_val_score(bagging_linear, X_train, y_train, cv = 5).mean()
# SVM RBF kernel
bagging_rbf = BaggingClassifier(base_estimator = SVC(kernel='rbf', random_state=1, gamma=0.02, C=1),n_estimators =24,random_state = 22)
bagging['SVM RBF kernel'] = cross_val_score(bagging_rbf, X_train, y_train, cv = 5).mean()
# SVM sigmoid kernel
bagging_sigmoid = BaggingClassifier(base_estimator = SVC(kernel='sigmoid'),n_estimators =24,random_state = 22)
bagging['SVM sigmoid kernel'] = cross_val_score(bagging_sigmoid, X_train, y_train, cv = 5).mean()


sorted_accuracy_bagging = [(k, bagging[k]) for k in sorted(bagging, key=bagging.get, reverse=True)]
print("Accuracy of each classifier follows below in sorted list:")
for k, v in sorted_accuracy_bagging:
    print("Model: {} has accuracy: {:.2%}".format(k,v))
