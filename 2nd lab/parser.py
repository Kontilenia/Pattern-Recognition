# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 17:57:49 2022

@author: User
"""

import os
from glob import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import nn
import torch
from torch.nn.utils.rnn import pad_packed_sequence



def parse_free_digits(directory):
    # Parse relevant dataset info
    files = glob(os.path.join(directory, "*.wav"))
    fnames = [f.split("\\")[1].split(".")[0].split("_") for f in files]
    ids = [f[2] for f in fnames]
    y = [int(f[0]) for f in fnames]
    speakers = [f[1] for f in fnames]
    _, Fs = librosa.core.load(files[0], sr=None)

    def read_wav(f):
        wav, _ = librosa.core.load(f, sr=None)

        return wav

    # Read all wavs
    wavs = [read_wav(f) for f in files]

    # Print dataset info
    print("Total wavs: {}. Fs = {} Hz".format(len(wavs), Fs))

    return wavs, Fs, ids, y, speakers


def extract_features(wavs, n_mfcc=6, Fs=8000):
    # Extract MFCCs for all wavs
    window = 30 * Fs // 1000
    step = Fs * 10 // 1000
    frames = [
        librosa.feature.mfcc(
            y=wav, sr=Fs, n_fft=window, hop_length=window - step, n_mfcc=n_mfcc
        ).T

        for wav in tqdm(wavs, desc="Extracting mfcc features...")
    ]

    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))

    return frames


def split_free_digits(frames, ids, speakers, labels):
    print("Splitting in train test split using the default dataset split")
    # Split to train-test
    X_train, y_train, spk_train = [], [], []
    X_test, y_test, spk_test = [], [], []
    test_indices = ["0", "1", "2", "3", "4"] 

    for idx, frame, label, spk in zip(ids, frames, labels, speakers):
        if str(idx) in test_indices:
            X_test.append(frame)
            y_test.append(label)
            spk_test.append(spk)
        else:
            X_train.append(frame)
            y_train.append(label)
            spk_train.append(spk)

    return X_train, X_test, y_train, y_test, spk_train, spk_test


def make_scale_fn(X_train):
    # Standardize on train data
    scaler = StandardScaler()
    scaler.fit(np.concatenate(X_train))
   # print("Normalization will be performed using mean: {}".format(scaler.mean_))
  #  print("Normalization will be performed using std: {}".format(scaler.scale_))
    def scale(X):
        scaled = []

        for frames in X:
            scaled.append(scaler.transform(frames))
        return scaled
    return scale


def parser(directory, n_mfcc=6):
    wavs, Fs, ids, y, speakers = parse_free_digits(directory)
    frames = extract_features(wavs, n_mfcc=n_mfcc, Fs=Fs)
    X_train, X_test, y_train, y_test, spk_train, spk_test = split_free_digits(
        frames, ids, speakers, y
    )

    return X_train, X_test, y_train, y_test, spk_train, spk_test

def predict(models, X):
        likelihoods = [m.score(X) for m in models]
        return np.argmax(likelihoods)

def accuracy(models, X, y):
    predictions = [predict(models, x) for x in X]
    assert len(predictions) == len(y)
    return sum(pred == true for pred, true in zip(predictions, y)) / len(y)


class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels):
        """
            feats: Python list of numpy arrays that contain the sequence features.
                   Each element of this list is a numpy array of shape seq_length x feature_dimension
            labels: Python list that contains the label for each sequence (each label must be an integer)
        """
        # self.lengths = len(feats[0]) # Find the lengths
        self.lengths = [i.shape[0] for i in feats]

        self.feats = self.zero_pad_and_stack(feats)
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels).astype('int64')

    def zero_pad_and_stack(self, x):
        """
            This function performs zero padding on a list of features and forms them into a numpy 3D array
            returns
                padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        padded = []
        # --------------- Insert your code here ---------------- #
        # padded = np.zeros((num_sequences, max_sequence_length, feature_dimension))
        max_sequence_length = max(self.lengths)
        padded = np.zeros((len(x),max_sequence_length,x[0].shape[1]))
        for idx,elem in enumerate(x):
            padded[idx,0:len(elem),:] = elem
        return padded

    def __getitem__(self, item):
        return self.feats[item], self.labels[item], self.lengths[item]

    def __len__(self):
        return len(self.feats)

# Custom LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim,cell_dim, layer_dim, output_dim, dropout_prob, bidirectional=False):
        super(LSTMModel, self).__init__()

        # 1 for not bidirectional and 2 for bidirectional
        self.D = 2 if bidirectional else 1
        # Number of hidden layers
        self.layer_dim = layer_dim
        # Hidden dimensions and hidden cells
        self.hidden_dim = hidden_dim
        self.cell_dim = cell_dim
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob,
                            bidirectional=bidirectional)
        # Linear layer
        self.fc = nn.Linear(int(self.D * hidden_dim), output_dim)

    def forward(self, x):

        # Read the maximum sequence length
        unpacked_x, unpacked_lengths = pad_packed_sequence(x, batch_first=True)
        # Hidden and cell state initializations
        h0 = torch.zeros(int(self.D * self.layer_dim), len(unpacked_lengths), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(int(self.D * self.layer_dim), len(unpacked_lengths), self.cell_dim).requires_grad_()
        # Forward propagation
        output, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # Unpacking output
        unpacked, unpacked_len = pad_packed_sequence(output, batch_first=True)
        output = unpacked
        # Index hidden state of last time step
        output = output[:, -1, :]
        # Output shape (batch_size, output_dim)
        output = self.fc(output)
        return output


if __name__ == "__main__":
    import sys
    
    #Step 9
    # parse data
    X_train, X_test, y_train, y_test, spk_train, spk_test = parser("recordings")
    print(len(X_train)+len(X_test))
    #we have about 3000 samples (50*10*6)
    # split data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train,random_state=200)
    # next, we normalize
    scale_fn = make_scale_fn(X_train)
    X_train = scale_fn(X_train)
    X_val = scale_fn(X_val)
    X_test = scale_fn(X_test)
    
    #Step 10

    from hmmlearn import hmm
    X = [] # data from a single digit (can be a numpy array)
    n_states = 4 # the number of HMM states
    n_mixtures = 5 # the number of Gaussians
    n_iter=8
    X_train_np = np.array(X_train,dtype='object')
    # X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    for i in range(10):
        X.append(X_train_np[y_train_np==i])
    #πρακτικά κάθε στοιχείο της λίστας περιέχει μία άλλη λίστα, όπου
    #κάθε στοιχείο της δεύτερης είναι πίνακας n_frames x 6 (<-mfcc)
    #για το κομματι του gmm και από σχόλιο στον github τα πετάω μέσα σα κουβά, δε με ενδιαφέρουν οι διαστάσεις
    digits_models = [{} for _ in range(10)]
    # digits_models: list[dict[tuple[int, int], hmm.GMMHMM]] = [{} for _ in range(10)]

    for d in range(10):
        for n_states in range(1, 5):
            for n_mix in range(1, 6):
                trans_mat = np.diag(np.ones(n_states)) + np.diag(np.ones(n_states - 1), 1)
                trans_mat = (trans_mat.T / np.sum(trans_mat, axis=1)).T
                starts = np.zeros(n_states)
                starts[0] = 1

                model = hmm.GMMHMM(
                    n_components=n_states,
                    n_mix=n_mix,
                    init_params="mcw",
                    params="tmcw",
                    n_iter=n_iter,
                    random_state=200
                )
                model.transmat_ = trans_mat
                model.startprob_ = starts
                digits_models[d][(n_states, n_mix)] = model
    #Step 11
    for d in range(10):
        for n_states in range(1, 5):
            for n_mix in range(1, 6):
                X_digit = X[d]
                lengths = [i.shape[0] for i in X_digit]
                digits_models[d][(n_states, n_mix)].fit(np.concatenate(X_digit), lengths)

    accuracies = np.zeros((4,5))
    for n_states in range(1, 5):
        for n_mix in range(1, 6):
            print(n_states, n_mix)
            accuracies[n_states-1,n_mix-1]=accuracy([d[(n_states, n_mix)] for d in digits_models], X_val, y_val)
            print(accuracy([d[(n_states, n_mix)] for d in digits_models], X_val, y_val))
    result = np.where(accuracies == np.amax(accuracies))
    print("Best model is:({},{})".format(result[0][0]+1,result[1][0]+1))
    print()
    print(result)


    #Step 13
    from sklearn.metrics import confusion_matrix
    from plot_confusion_matrix import plot_confusion_matrix

    model = [d[(result[0][0]+1,result[1][0]+1)] for d in digits_models]

    preds = [predict(model, x) for x in X_val]
    cm = confusion_matrix(y_val, preds)
    plot_confusion_matrix(cm, list(range(10)), normalize=True)
    print("Next confusion:")
    preds = [predict(model, x) for x in X_test]
    cm = confusion_matrix(y_test, preds)
    plot_confusion_matrix(cm, list(range(10)), normalize=True)

    batch_size = 64

    # step 14
    train_features, train_targets = torch.Tensor(X_train), torch.Tensor(y_train)
    test_features, test_targets = torch.Tensor(X_test), torch.Tensor(y_test)
    val_features, val_targets = torch.Tensor(X_val), torch.Tensor(y_val)

    train = TensorDataset(train_features, train_targets)
    test = TensorDataset(test_features, test_targets)
    val = TensorDataset(val_features, val_targets)
