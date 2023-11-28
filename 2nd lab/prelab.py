import os
from glob import glob

import librosa
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import matplotlib.axes as axs
import random
import pandas as pd
from word2number import w2n
import seaborn as sn
import scipy.stats as sts
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def parse_free_digits(directory):
    # Parse relevant dataset info
    files = glob(os.path.join(directory, "*.wav"))
    f_names2 = [[f.split("\\")[1].split(".")[0]] for f in files]
    all_splits = [split_name_number(i[0]) for i in f_names2]
    digits, speakers = list(map(list, zip(*all_splits)))
    digits = [w2n.word_to_num(i) for i in digits]
    ids = [i for i in range(len(speakers))]
    _, Fs = librosa.core.load(files[0], sr=None)

    def read_wav(f):
        wav, _ = librosa.core.load(f, sr=None)
        return wav

    # Read all wavs
    wavs = [read_wav(f) for f in files]

    # Print dataset info
    print("Total wavs: {}. Fs = {} Hz".format(len(wavs), Fs))

    return wavs, Fs, ids, digits, speakers


def split_name_number(s):
    head = s.rstrip('0123456789')  # at the end cut the numbers
    tail = int(s[len(head):])  # whatever remains
    return head, tail


def extract_features(wavs, n_mfcc=13, Fs=16000):
    # Extract MFCCs for all wavs
    window = Fs * 25 // 1000
    step = Fs * 10 // 1000
    mfcc_frames = [
        librosa.feature.mfcc(
            y=wav, sr=Fs, n_fft=window, hop_length=window - step, n_mfcc=n_mfcc
        )

        for wav in tqdm(wavs, desc="Extracting mfcc features...")
    ]
    deltas_frames = [
        librosa.feature.delta(mfcc_f)

        for mfcc_f in tqdm(mfcc_frames, desc="Extracting delta features...")
    ]
    delta_deltas_frames = [
        librosa.feature.delta(mfcc_f, order=2)

        for mfcc_f in tqdm(mfcc_frames, desc="Extracting delta deltas features...")
    ]
    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))
    return mfcc_frames, deltas_frames, delta_deltas_frames


# def split_free_digits(frames, ids, speakers, labels):
#     print("Splitting in train test split using the default dataset split")
#     # Split to train-test
#     X_train, y_train, spk_train = [], [], []
#     X_test, y_test, spk_test = [], [], []
#     test_indices = ["0", "1", "2", "3", "4"]
#
#     for idx, frame, label, spk in zip(ids, frames, labels, speakers):
#         if str(idx) in test_indices:
#             X_test.append(frame.T)
#             y_test.append(label)
#             spk_test.append(spk)
#         else:
#             X_train.append(frame.T)
#             y_train.append(label)
#             spk_train.append(spk)
#
#     return X_train, X_test, y_train, y_test, spk_train, spk_test


# def make_scale_fn(X_train):
#     # Standardize on train data
#     scaler = StandardScaler()
#     # scaler.fit(np.concatenate(X_train))
#     scaler.fit(X_train)
#     print("Normalization will be performed using mean: {}".format(scaler.mean_))
#     print("Normalization will be performed using std: {}".format(scaler.scale_))
#
#     def scale(X):
#         scaled = []
#
#         for frames in X:
#             scaled.append(scaler.transform(frames))
#         return scaled
#
#     return scale

def normalize_data(X):
    return normalize(X,axis=0, norm='max')

def parser(directory, n_mfcc=13):
    wavs, Fs, ids, y, speakers = parse_free_digits(directory)
    mfccs, deltas, delta_deltas = extract_features(wavs, n_mfcc=n_mfcc, Fs=Fs)
    return mfccs, deltas, delta_deltas, wavs, speakers, ids, y, Fs


def find_digits(all_digits, n1, n2):
    n1_indexes = []
    n2_indexes = []
    for i in range(len(all_digits)):
        if all_digits[i] == n1:
            n1_indexes.append(i)
        if all_digits[i] == n2:
            n2_indexes.append(i)
    # Same thing but in two lines
    # n1_indexes2 = [i for i in range(len(all_digits)) if all_digits[i] == n1]
    # n2_indexes2 = [i for i in range(len(all_digits)) if all_digits[i] == n2]
    return n1_indexes, n2_indexes


def plot_hist(mfcc, n1_list, n2_list):
    n1_0 = mfcc[n1_list[0]][:][0]  # from each n1 takes the first coefficient
    n1_1 = mfcc[n1_list[0]][:][1]  # from each n1 takes the second coefficient
    n2_0 = mfcc[n2_list[0]][:][0]  # from each n2 takes the first coefficient
    n2_1 = mfcc[n2_list[0]][:][1]  # from each n2 takes the second coefficient
    for i in range(1, len(n1_list)):
        n1_0 = np.concatenate((n1_1, mfcc[n1_list[i]][:][0]),
                              axis=0)  # make a union array with all features of the first coefficients of n1
        n1_1 = np.concatenate((n1_1, mfcc[n1_list[i]][:][1]),
                              axis=0)  # make a union array with all features of the second coefficients of n1
    for i in range(1, len(n2_list)):
        n2_0 = np.concatenate((n2_0, mfcc[n2_list[i]][:][0]),
                              axis=0)  # make a union array with all features of the first coefficients of n2
        n2_1 = np.concatenate((n2_1, mfcc[n2_list[i]][:][1]),
                              axis=0)  # make a union array with all features of the second coefficients of n2
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
    ax0.hist(n1_0.T, histtype='bar')
    ax0.set_title('Four with 1st coefficient')
    ax1.hist(n1_1.T, histtype='bar')
    ax1.set_title('Four with 2nd coefficient')
    ax2.hist(n2_0.T, histtype='bar')
    ax2.set_title('Seven with 1st coefficient')
    ax3.hist(n2_1.T, histtype='bar')
    ax3.set_title('Seven with 2nd coefficient')
    fig.tight_layout()
    plt.show()


# def digits_and_speakers(n1_list, n2_list, speakers):
def digits_and_speakers(n1_list, n2_list):
    # speaker1_n1 = random.randint(min(n1_list), max(n1_list))
    # speaker2_n1 = random.randint(min(n1_list), max(n1_list))
    # while (speakers[speaker1_n1] == speakers[speaker2_n1]):
    #     speaker1_n2 = random.randint(min(ids), max(ids))
    # speaker1_n2 = random.randint(min(n2_list), max(n2_list))
    # speaker2_n2 = random.randint(min(n2_list), max(n2_list))
    # while (speakers[speaker1_n2] == speakers[speaker2_n2]):
    #     speaker1_n2 = random.randint(min(ids), max(ids))
    random.seed(10)
    speaker1_n1, speaker2_n1 = random.sample(n1_list, 2)
    speaker1_n2, speaker2_n2 = random.sample(n2_list, 2)
    return speaker1_n1, speaker1_n2, speaker2_n1, speaker1_n2


def melspecrogram(wavs, Fs, window, step):
    melspec = librosa.feature.melspectrogram(y=wavs, sr=Fs, n_fft=window, hop_length=window - step, n_mels=13)
    return melspec


def scatter(x, y, grouper, title=None, ax=None, xlabel=None, ylabel=None, ):
    sn.scatterplot(x=x, y=y, hue=grouper, style=grouper, palette='cividis_r', legend='full', ax=ax, )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def analysis(mfccs, deltas, deltas2, digits):
    # mfccs, deltas, deltas2 = extract_features(wavs, n_mfcc=13, Fs=16000)
    # meancon = []
    # stdcon = []
    # unique = []
    features = np.zeros((len(wavs), 78))
    # for i in range(133):
    for i in range(len(wavs)):  # more general code
        features[i, :] = np.concatenate(
            (mfccs[i].mean(axis=1), deltas[i].mean(axis=1), deltas2[i].mean(axis=1), mfccs[i].std(axis=1),
             deltas[i].std(axis=1), deltas2[i].std(axis=1)), axis=0)
        # a = (mfccs[i].mean(axis=1)).tolist() + (deltas[i].mean(axis=1)).tolist() + (deltas2[i].mean(axis=1)).tolist()
        # meancon.append(a)
        # b = (mfccs[i].std(axis=1)).tolist() + (deltas[i].std(axis=1)).tolist() + (deltas2[i].std(axis=1)).tolist()
        # stdcon.append(b)
        # c = a + b
        # unique.append(c)
        # πίνακας 133 σειρών όπου κάθε σειρά αντιπροσωπεύει μία εκφώνηση
        # και κάθε σειρά αποτελείται από μία 39 στήλες: μέση τιμή των 13 mfcc + deltas +deltas2
    # fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    # scatter(np.asarray(meancon)[:, 0], np.asarray(meancon)[:, 1], digits, 'Means', ax=axs[0])
    # scatter(np.asarray(stdcon)[:, 0], np.asarray(stdcon)[:, 1], digits, 'Standard Deviations', ax=axs[1])
    # plt.show()
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    scatter(features[:, 0], features[:, 1], digits, 'Means', ax=axs[0])
    scatter(features[:, 39], features[:, 40], digits, 'Standard Deviations', ax=axs[1])
    plt.show()
    # return meancon, stdcon, unique
    return features


def pca_analysis(meancon, digits):
    pca = PCA(n_components=2)
    new_mean = pca.fit_transform(meancon)
    fig, axs = plt.subplots(ncols=1, figsize=(6, 4))
    scatter(np.asarray(new_mean)[:, 0], np.asarray(new_mean)[:, 1], digits, 'Means', ax=axs)
    #  scatter(np.asarray(new_std)[:, 0], np.asarray(new_std)[:, 1], digits, 'Standard Deviations', ax=axs[1])
    plt.show()
    print("The original variance retained using 2-component PCA is {:.2%}".format(
        pca.explained_variance_ratio_.cumsum()[-1]))
    return new_mean


def pca_analysis_3d(meancon, digits, text):
    # meancon,stdcon,n,digits
    # reduced_m, evr_m, reduced_s, evr_s,digits
    pca = PCA(n_components=3)
    new_mean = pca.fit_transform(meancon)
    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    # Creating plot
    scatter = ax.scatter3D(np.asarray(new_mean)[:, 0], np.asarray(new_mean)[:, 1], c=digits, cmap='cividis_r')
    plt.title("3D scatter plot of " + text)
    legend_m = ax.legend(*scatter.legend_elements(), loc="upper left", title="Digits")
    ax.add_artist(legend_m)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    # show plot
    plt.show()
    print("The original variance retained using 3-component PCA is {:.2%}".format(
        pca.explained_variance_ratio_.cumsum()[-1]))
    return new_mean


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
        if not isinstance(y, np.ndarray):
            y = y.to_numpy()
        for i in range(0, 9):
            # mean_variables = np.mean(X[y == i + 1], axis=0)
            mean_variables = np.mean(X[np.where(y == i + 1)[0],:], axis=0)
            mean_value.append(mean_variables)

            if nb_var is None:
                # var_variables = np.var(X[y == i + 1], axis=0)
                var_variables = np.var(X[np.where(y == i + 1)[0],:], axis=0)
                var_value.append(var_variables)
            else:
                var_value.append(np.ones(78) * nb_var)
            # apriori_value.append(len(y[y == i + 1]) / len(y))
            apriori_value.append(len(y[y == i + 1]) / len(y))

        var_value = np.array(var_value)

        # print(var_value)
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
        # to compare it with 1-9 digits
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        for i in range(X.shape[0]):
            for k in range(0, 9):
                a = 1
                for j in range(X.shape[1]):
                    a = a * calculate_probability(X[i, j], self.X_mean_[k, j], self.X_var_[k, j])
                probabilities.append(a * self.X_apriori_[k])
            predictions.append(np.argmax(probabilities) + 1)
            probabilities.clear()
        return pd.DataFrame(predictions)

    def score(self, X, y, sample_weight=None):

        pred = self.predict(X)
        pred = np.ravel(pred.to_numpy())
        if not isinstance(y, np.ndarray):
            y = y.to_numpy()
        return len(y[y == pred]) / len(y)


if __name__ == "__main__":
    # Step 2 and 3
    mfccs_all, deltas_all, delta_deltas_all, wavs, speakers, ids, all_digits, Fs = parser("digits")

    # step 4
    mfcs = []  # list for Mel Filter-bank Spectral Coefficients
    mfcc = []  # list for Mel-Frequency Cepstral Coefficients
    n1 = 4  # define n1
    n2 = 7  # define n2
    n1_list, n2_list = find_digits(all_digits, n1, n2)  # func to find the indexes of all n1 and n2
    plot_hist(mfccs_all, n1_list, n2_list)

    # speaker1_n1, speaker1_n2, speaker2_n1, speaker2_n2 = digits_and_speakers(n1_list, n2_list, speakers)
    speaker1_n1, speaker1_n2, speaker2_n1, speaker2_n2 = digits_and_speakers(n1_list, n2_list)
    speak = [speaker1_n1, speaker1_n2, speaker2_n1, speaker2_n2]
    # MFSC calculations and heatmaps
    window = Fs * 25 // 1000
    step = Fs * 10 // 1000
    for i in speak:
        mfcs.append(melspecrogram(wavs[i], Fs, window, step))
        mfcc.append(mfccs_all[i])
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
    ax1 = [ax0, ax1, ax2, ax3]
    for i in range(len(ax1)):
        sn.heatmap(np.corrcoef(mfcs[i]), cmap='viridis', ax=ax1[i])
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
    ax = [ax0, ax1, ax2, ax3]
    for i in range(len(ax)):
        sn.heatmap(np.corrcoef(mfcc[i]), cmap='viridis', ax=ax[i])
    plt.show()

    # Step 5
    # meancon, stdcon, mean_std = analysis(wavs, all_digits)
    features = analysis(mfccs_all, deltas_all, delta_deltas_all, all_digits)

    # step 6
    # mean2 = pca_analysis(mean_con, all_digits)
    # std2 = pca_analysis(std_con, all_digits)
    # mean3 = pca_analysis_3d(mean_con, all_digits, 'means')
    # std3 = pca_analysis_3d(std_con, all_digits, 'standard deviations')

    mean2 = pca_analysis(features[:, 0:39], all_digits)
    std2 = pca_analysis(features[:, 39:], all_digits)
    mean3 = pca_analysis_3d(features[:, 0:39], all_digits, 'means')
    std3 = pca_analysis_3d(features[:, 39:], all_digits, 'standard deviations')

    # step 7
    # X = pd.DataFrame(mean_std)
    # X = pd.DataFrame(features)
    # y = pd.DataFrame(all_digits)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7)
    X_train, X_test, y_train, y_test = train_test_split(np.array(features), np.array(all_digits), test_size=0.3, train_size=0.7,random_state=2000) # scikit learns works better with np
    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)
    Bayes = NaiveBayesClassifier()
    Bayes = Bayes.fit(X_train, y_train)
    Bayes_score = Bayes.score(X_test, y_test)
    print("Success Performance of our custom Naive Bayes classifier is: {:.2%}".format(Bayes_score))

    # X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train)
    # print("If using all data to calculate normalization statistics")
    # scale_fn = make_scale_fn(X_train + X_dev + X_test)
    # print("If using X_train + X_dev to calculate normalization statistics")
    # scale_fn = make_scale_fn(X_train + X_dev)
    # print("If using X_train to calculate normalization statistics")
    # scale_fn = make_scale_fn(X_train)
    # X_train = scale_fn(X_train)
    # X_dev = scale_fn(X_dev)
    # X_test = scale_fn(X_test)

    classifiers = {}
    # Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    classifiers['Naive Bayes'] = gnb.score(X_test, y_test)
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
    # Decision Tree Classifier
    decision_tree = DecisionTreeClassifier()

    sorted_accuracy = [(k, classifiers[k]) for k in sorted(classifiers, key=classifiers.get, reverse=True)]
    print("Success of each scikit-learn classifier follows below in sorted list:")
    for k, v in sorted_accuracy:
        print("Model: {} has performance: {:.2%}".format(k, v))
