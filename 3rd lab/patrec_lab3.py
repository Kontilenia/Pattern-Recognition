# -*- coding: utf-8 -*-
"""patrec-lab3.ipynb

**Step 0**
"""

!git clone https://github.com/slp-ntua/patrec-labs.git

import sys
sys.path.append('/kaggle/working/patrec-labs/lab3')

# Import data
import os
os.listdir('../input/patreco3-multitask-affective-music/data/')

"""**Step 1, 2a & 3 for mel spectrograms and chromagrams**"""

import random
import pandas as pd
import numpy as np
import torch as t
import sklearn as sk

data = pd.read_csv('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train_labels.txt', sep="\t")

# Select and load 2 random files with different genre
random_samples = list(data.set_index('Genre').sample(n=2,axis=0,random_state=42)["Id"])

prefix = '/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train/'

sample1 = prefix + random_samples[0][:-3]
sample2 = prefix + random_samples[1][:-3]

spec1 = np.load(sample1)
spec2 = np.load(sample2)

# Decompose  mel spectrograms and chromagrams from the files
mel1, chroma1 = spec1[:128], spec1[128:]
mel2, chroma2 = spec2[:128], spec2[128:]

attributes = [(mel1,chroma1),(mel2,chroma2)]

import librosa.display
import matplotlib.pyplot as plt

# Plot spectograms and chromagrams
for mel,chroma in attributes:
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mel, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title='Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    fig, ax = plt.subplots()
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    ax.set(title='Chromagram')
    fig.colorbar(img, ax=ax)

print("Dimensions of spectogram 1 are: {}".format(mel1.shape))
print("Dimensions of spectogram 2 are: {}".format(mel2.shape))
print("Dimensions of chromagram 1 are: {}".format(chroma1.shape))
print("Dimensions of chromagram 2 are: {}".format(chroma2.shape))

"""**Step 2b & 3 for beat-synced spectrograms spectrograms and chromagrams**"""

prefix = '/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/train/'

# Select and load 2 from the previous random files
sample3 = prefix + random_samples[0][:-3]
sample4 = prefix + random_samples[1][:-3]

spec3 = np.load(sample3)
spec4 = np.load(sample4)

# Decompose  mel spectrograms and chromagrams from the files
mel3, chroma3 = spec3[:128], spec3[128:]
mel4, chroma4 = spec4[:128], spec4[128:]
attributes2 = [(mel3,chroma3),(mel4,chroma4)]

# Plot spectograms and chromagrams
for mel,chroma in attributes2:
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mel, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title='Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    fig, ax = plt.subplots()
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    ax.set(title='Chromagram')
    fig.colorbar(img, ax=ax)

print("Dimensions of beat-synced spectogram 1 are: {}".format(mel3.shape))
print("Dimensions of beat-synced spectogram 2 are: {}".format(mel4.shape))
print("Dimensions of beat-synced chromagram 1 are: {}".format(chroma3.shape))
print("Dimensions of beat-synced chromagram 2 are: {}".format(chroma4.shape))

"""**Step 4 a & b**"""

import copy
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

# Class mapping to merge similar classes and ignore classes that do not work very well
CLASS_MAPPING = {
    "Rock": "Rock",
    "Psych-Rock": "Rock",
    "Indie-Rock": None,
    "Post-Rock": "Rock",
    "Psych-Folk": "Folk",
    "Folk": "Folk",
    "Metal": "Metal",
    "Punk": "Metal",
    "Post-Punk": None,
    "Trip-Hop": "Trip-Hop",
    "Pop": "Pop",
    "Electronic": "Electronic",
    "Hip-Hop": "Hip-Hop",
    "Classical": "Classical",
    "Blues": "Blues",
    "Chiptune": "Electronic",
    "Jazz": "Jazz",
    "Soundtrack": None,
    "International": None,
    "Old-Time": None,
}


# Return mel spectrograms or chromagrams from a file
def read_spectrogram(spectrogram_file, feat_type):
    spectrogram = np.load(spectrogram_file)
    # spectrograms contains a fused mel spectrogram and chromagram
    
    if feat_type=='mel':
        return spectrogram[:128, :].T
    elif feat_type=='chroma':
        return spectrogram[128:, :].T

    return spectrogram.T


# Label Tranformation
class LabelTransformer(LabelEncoder):
    def inverse(self, y):
        try:
            return super(LabelTransformer, self).inverse_transform(y)
        except:
            return super(LabelTransformer, self).inverse_transform([y])

    def transform(self, y):
        try:
            return super(LabelTransformer, self).transform(y)
        except:
            return super(LabelTransformer, self).transform([y])

        
# Padding using max_length 
class PaddingTransform(object):
    def __init__(self, max_length, padding_value=0):
        self.max_length = max_length
        self.padding_value = padding_value

    def __call__(self, s):
        if len(s) == self.max_length:
            return s

        if len(s) > self.max_length:
            return s[: self.max_length]

        if len(s) < self.max_length:
            s1 = copy.deepcopy(s)
            pad = np.zeros((self.max_length - s.shape[0], s.shape[1]), dtype=np.float32)
            s1 = np.vstack((s1, pad))
            return s1

# Class for our Dataset
class SpectrogramDataset(Dataset):
    def __init__(
        self, path, class_mapping=None, train=True, feat_type='mel', max_length=-1, regression=None
    ):
        t = "train" if train else "test"
        p = os.path.join(path, t)
        self.regression = regression

        self.full_path = p
        self.index = os.path.join(path, "{}_labels.txt".format(t))
        self.files, labels = self.get_files_labels(self.index, class_mapping)
        self.feats = [read_spectrogram(os.path.join(p, f), feat_type) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)
        self.label_transformer = LabelTransformer()
        if isinstance(labels, (list, tuple)):
            if not regression:
                self.labels = np.array(
                    self.label_transformer.fit_transform(labels)
                ).astype("int64")
            else:
                self.labels = np.array(labels).astype("float64")

    def get_files_labels(self, txt, class_mapping):
        with open(txt, "r") as fd:
            lines = [l.rstrip().split("\t") for l in fd.readlines()[1:]]
        files, labels = [], []
        for l in lines:            
            if self.regression:
                l = l[0].split(",")
                files.append(l[0] + ".fused.full.npy")
                labels.append(l[self.regression])
                continue
            label = l[1]
            if class_mapping:
                label = class_mapping[l[1]]
            if not label:
                continue
            fname = l[0]
            if fname.endswith(".gz"):
                fname = ".".join(fname.split(".")[:-1])
            
            if 'fma_genre_spectrograms_beat' in self.full_path.split('/'): # necessary fix 1
                fname = fname.replace('beatsync.fused', 'fused.full')            
            if 'test' in self.full_path.split('/'): # necessary fix 2
                fname = fname.replace('full.fused', 'fused.full')
            
            files.append(fname)
            labels.append(label)
        return files, labels

    def __getitem__(self, item):
        length = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), self.labels[item], length

    def __len__(self):
        return len(self.labels)
    

# Plot histogram for class and frequencies
def plot_class_histogram(dataset, title):
    plt.figure()
    plt.hist(dataset.labels, rwidth=0.5, align='mid')
    plt.xticks(dataset.labels, dataset.labels.astype(str))
    plt.title(title)
    plt.xlabel('Class Labels')
    plt.ylabel('Frequencies')

"""**Step 4 c**"""

PARENT_DATA_DIR = '../input/patreco3-multitask-affective-music/data/'

# Merged classes dataset
train_dataset = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, 'fma_genre_spectrograms'), class_mapping=CLASS_MAPPING, train=True
)

plot_class_histogram(train_dataset, title='Histogram of classes with merged classes')

# All classes dataset
train_dataset_all = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, 'fma_genre_spectrograms'), class_mapping=None, train=True
)

plot_class_histogram(train_dataset_all, title='Histogram of classes with all classes')

import gc

# Release memory from RAM
del train_dataset
del train_dataset_all

# Call garbage collector
gc.collect()

"""**Step 5 a**"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from lstm import LSTMBackbone

# General class for classifier implemenation
class Classifier(nn.Module):
    def __init__(self, num_classes, backbone, load_from_checkpoint=None):
        """
        backbone (nn.Module): The nn.Module to use for spectrogram parsing
        num_classes (int): The number of classes
        load_from_checkpoint (Optional[str]): Use a pretrained checkpoint to initialize the model
        """
        super(Classifier, self).__init__()
        self.backbone = backbone  # An LSTMBackbone or CNNBackbone
        if load_from_checkpoint is not None:
            self.backbone = load_backbone_from_checkpoint(
                self.backbone, load_from_checkpoint
            )
        self.is_lstm = isinstance(self.backbone, LSTMBackbone)
        self.output_layer = nn.Linear(self.backbone.feature_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()  # Loss function for classification

    def forward(self, x, targets, lengths):
        feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)
        logits = self.output_layer(feats)
        loss = self.criterion(logits, targets)
        return loss, logits

"""**Step 5 b**"""

# Custom function for train and validation split for current datasets
def torch_train_val_split(
    dataset, batch_train, batch_eval, val_size=0.2, shuffle=True, seed=420
):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_size * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_train, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_eval, sampler=val_sampler)
    return train_loader, val_loader

BATCH_SIZE = 8
MAX_LENGTH = 150
DEVICE = 'cuda'

# Dataset for mel sprectrogram
train_dataset = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, 'fma_genre_spectrograms'), class_mapping=CLASS_MAPPING, 
    train=True, feat_type='mel', max_length=MAX_LENGTH
)

# Dataloaders for train and validation set
train_loader, val_loader = torch_train_val_split(train_dataset, BATCH_SIZE, BATCH_SIZE)

# get the 1st batch values of the data loader
x_b1, y_b1, lengths_b1 = next(iter(train_loader))

# print the shape of the 1st item of the 1st batch of the data loader
input_shape = x_b1[0].shape
print(input_shape)

del train_dataset
gc.collect()

def overfit_with_a_couple_of_batches(model, train_loader, optimizer, device):
    print('Training in overfitting mode...')
    epochs = 400
    
    # get only the 1st batch
    x_b1, y_b1, lengths_b1 = next(iter(train_loader))    
    model.train()
    for epoch in range(epochs):        
        loss, logits = model(x_b1.float().to(device), y_b1.to(device), lengths_b1.to(device))
        # prepare
        optimizer.zero_grad()
        # backward
        loss.backward()
        # optimizer step
        optimizer.step()

        if epoch == 0 or (epoch+1)%20 == 0:
            print(f'Epoch {epoch+1}, Loss at training set: {loss.item()}')
            
def train(model, train_loader, val_loader, optimizer, epochs, device="cuda", overfit_batch=False):
    if overfit_batch:
        overfit_with_a_couple_of_batches(model, train_loader, optimizer, device)
    else:
        pass

DEVICE = 'cuda'

# Training LSTM in overfitting mode
RNN_HIDDEN_SIZE = 64
NUM_CATEGORIES = 10

LR = 1e-4
epochs = 10

backbone = LSTMBackbone(input_shape[1], rnn_size=RNN_HIDDEN_SIZE, num_layers=2, bidirectional=True)
model = Classifier(NUM_CATEGORIES, backbone)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train(model, train_loader, val_loader, optimizer, epochs, device=DEVICE, overfit_batch=True)

"""**Step 5 c,d,e,f**"""

# EarlyStopping adopted from: https://stackoverflow.com/a/73704579/19306080 and additional customizations
class EarlyStopper:
    def __init__(self, model, save_path, patience=1, min_delta=0):
        self.model = model
        self.save_path = save_path
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(self.model.state_dict(), self.save_path)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# Traing for one epoch 
def train_one_epoch(model, train_loader, optimizer, device=DEVICE):
    model.train()
    total_loss = 0
    for x, y, lengths in train_loader:        
        loss, logits = model(x.float().to(device), y.to(device), lengths.to(device))
        # prepare
        optimizer.zero_grad()
        # backward
        loss.backward()
        # optimizer step
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)    
    return avg_loss

# Validate for one epoch 
def validate_one_epoch(model, val_loader, device=DEVICE):    
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y, lengths in val_loader:
            loss, logits = model(x.float().to(device), y.to(device), lengths.to(device))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)    
    return avg_loss

# Implemetation of the training process with and witout overfitting
def train(model, train_loader, val_loader, optimizer, epochs, save_path='checkpoint.pth', device="cuda", overfit_batch=False):
    if overfit_batch:
        overfit_with_a_couple_of_batches(model, train_loader, optimizer, device)
    else:
        print(f'Training started for model {save_path.replace(".pth", "")}...')
        early_stopper = EarlyStopper(model, save_path, patience=5)
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer)
            validation_loss = validate_one_epoch(model, val_loader)
            if epoch== 0 or (epoch+1)%5==0:
                print(f'Epoch {epoch+1}/{epochs}, Loss at training set: {train_loss}\n\tLoss at validation set: {validation_loss}')          
            
            if early_stopper.early_stop(validation_loss):
                print('Early Stopping was activated.')
                print(f'Epoch {epoch+1}/{epochs}, Loss at training set: {train_loss}\n\tLoss at validation set: {validation_loss}')
                print('Training has been completed.\n')
                break

# LSTM hyperparametres, dataset paths and model saving details
BATCH_SIZE = 8
MAX_LENGTH = 150
DEVICE = 'cuda'
RNN_HIDDEN_SIZE = [64,128]
NUM_CATEGORIES = 10
LR = 1e-4
epochs =[10,40]
save_paths = ['lstm_genre_mel.pth','lstm_genre_beat.pth','lstm_genre_chroma.pth','lstm_genre_chroma_beat.pth','lstm_genre_all.pth','lstm_genre_all_beat.pth']
path = ['fma_genre_spectrograms','fma_genre_spectrograms_beat','fma_genre_spectrograms','fma_genre_spectrograms_beat','fma_genre_spectrograms','fma_genre_spectrograms_beat']
feat_types =['mel','mel','chroma','chroma','','']
input_shape = [128,128,12,12,140,140]

# Total LSTM training for all datasets (creating 6 different models)
from lstm import LSTMBackbone

for i,p in enumerate(path):
    train_dataset = SpectrogramDataset(
        os.path.join(PARENT_DATA_DIR, p), class_mapping=CLASS_MAPPING, 
        train=True, feat_type=feat_types[i], max_length=MAX_LENGTH
    )

    train_loader, val_loader = torch_train_val_split(train_dataset, BATCH_SIZE, BATCH_SIZE)

    # Run training
    backbone = LSTMBackbone(input_shape[i], rnn_size=RNN_HIDDEN_SIZE[1], num_layers=2, bidirectional=True)
    model = Classifier(NUM_CATEGORIES, backbone)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Run training in overfitting mode
    # train(model, train_loader, val_loader, optimizer, epochs[0], device=DEVICE, overfit_batch=True)
    train(model, train_loader, val_loader, optimizer, epochs[1], save_path=save_paths[i], device=DEVICE, overfit_batch=False)
    
    # Release memory from RAM
    del model
    del train_loader
    del val_loader
    del train_dataset

    # Call garbage collector
    gc.collect()

from sklearn.metrics import accuracy_score, classification_report

# Test function
def test(model, test_dataset, path, batch_size, device=DEVICE):   
    
    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model.load_state_dict(torch.load(path))
    model.eval()
    
    predicted = []
    ground_truth = []
    
    # Find true and predicted labels
    with torch.no_grad():
        for x, y, lengths in test_loader:
            _, logits = model(x.float().to(device), y.to(device), lengths.to(device))
            predicted.append(torch.max(logits.to("cpu"), 1)[1].tolist())
            ground_truth.append(y.tolist())
            
    predicted = [item for sublist in predicted for item in sublist]
    ground_truth = [item for sublist in ground_truth for item in sublist]
    
    # Calculate accuracy, percision, recall, F1-score    
    print(classification_report(np.array(predicted), np.array(ground_truth), zero_division=0))
    return accuracy_score(np.array(predicted), np.array(ground_truth))

OUTPUT_DATA_DIR = '/kaggle/working/'

# Total testing for all LSTM models (for 6 different models)
for i,p in enumerate(path):
    test_dataset = SpectrogramDataset(
        os.path.join(PARENT_DATA_DIR, p), class_mapping=CLASS_MAPPING, 
        train=False, feat_type=feat_types[i], max_length=MAX_LENGTH
    )
    
    backbone = LSTMBackbone(input_shape[i], rnn_size=RNN_HIDDEN_SIZE[1], num_layers=2, bidirectional=True)
    model = Classifier(NUM_CATEGORIES, backbone) 
    
    model.to(DEVICE)
    print("For {} model:".format(save_paths[i]))
    print(f"Accuracy: {test(model, test_dataset, os.path.join(OUTPUT_DATA_DIR,save_paths[i]), BATCH_SIZE)}",'\n')
    
    del model    
    gc.collect()

"""**Step 7**"""

# 2D CNN from convolution.py with additive implementation
class CNNBackbone(nn.Module):
    def __init__(self, input_dims, in_channels, filters, feature_size):
        super(CNNBackbone, self).__init__()
        self.input_dims = input_dims
        self.in_channels = in_channels
        self.filters = filters
        self.feature_size = feature_size
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, filters[0], kernel_size=(5,5), stride=1, padding=2),
            nn.BatchNorm2d((self.in_channels**1) * filters[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv2 = nn.Sequential(
                nn.Conv2d(filters[0], filters[1], kernel_size=(5,5), stride=1, padding=2),
                nn.BatchNorm2d((self.in_channels**2) * filters[1]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv3 = nn.Sequential(
                nn.Conv2d(filters[1], filters[2], kernel_size=(3,3), stride=1, padding=1),
                nn.BatchNorm2d((self.in_channels**3) * filters[2]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv4 = nn.Sequential(
                nn.Conv2d(filters[2], filters[3], kernel_size=(3,3), stride=1, padding=1),
                nn.BatchNorm2d((self.in_channels**4) * filters[3]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        
        shape_after_convs = [input_dims[0]//2**(len(filters)), input_dims[1]//2**(len(filters))]
        self.fc1 = nn.Linear(filters[3] * shape_after_convs[0] * shape_after_convs[1], self.feature_size)
        
    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, x.shape[1], x.shape[2])
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out

# CNN hyperparametres, dataset paths and model saving details
BATCH_SIZE = 8
MAX_LENGTH = 150
DEVICE = 'cuda'
NUM_CATEGORIES = 10
cnn_in_channels = 1
cnn_filters = [32, 64, 128, 256]
cnn_out_feature_size = 1000
LR = 1e-4
epochs = [10, 40]
save_paths_cnn = ['cnn_genre_mel.pth','cnn_genre_beat.pth',]
path_cnn = ['fma_genre_spectrograms','fma_genre_spectrograms_beat']
feat_types_cnn =['mel','mel']
input_shape_cnn = torch.Size([150, 128])

#  Training in overfitting mode for the CNN 
train_dataset = SpectrogramDataset(
        os.path.join(PARENT_DATA_DIR, path_cnn[0]), class_mapping=CLASS_MAPPING, 
        train=False, feat_type=feat_types_cnn[0], max_length=MAX_LENGTH
    )

train_loader, val_loader = torch_train_val_split(train_dataset, BATCH_SIZE, BATCH_SIZE)
    
# get the input shape
x_b1, y_b1, lengths_b1 = next(iter(train_loader))
input_shape = x_b1[0].shape

backbone = CNNBackbone(input_shape, cnn_in_channels, cnn_filters, cnn_out_feature_size)
model = Classifier(NUM_CATEGORIES, backbone)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train(model, train_loader, val_loader, optimizer, epochs[0], device=DEVICE, overfit_batch=True)

# Total CNN training for mel spectrograms in the datasets (creating 2 different models)
for i,p in enumerate(path_cnn):
    train_dataset = SpectrogramDataset(
        os.path.join(PARENT_DATA_DIR, p), class_mapping=CLASS_MAPPING, 
        train=True, feat_type=feat_types_cnn[i], max_length=MAX_LENGTH
    )

    train_loader, val_loader = torch_train_val_split(train_dataset, BATCH_SIZE, BATCH_SIZE)

    # Run training
    backbone = CNNBackbone(input_shape_cnn, cnn_in_channels, cnn_filters, cnn_out_feature_size)
    model = Classifier(NUM_CATEGORIES, backbone)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Run training in overfitting mode
    # train(model, train_loader, val_loader, optimizer, epochs[0], device=DEVICE, overfit_batch=True)
    train(model, train_loader, val_loader, optimizer, epochs[1], save_path=save_paths_cnn[i], device=DEVICE, overfit_batch=False)
    
    # Release memory from RAM
    del model
    del train_loader
    del val_loader
    del train_dataset

    # Call garbage collector
    gc.collect()

# Total CNN testing for all models (for 2 different models)
for i,p in enumerate(path_cnn):
    test_dataset = SpectrogramDataset(
        os.path.join(PARENT_DATA_DIR, p), class_mapping=CLASS_MAPPING, 
        train=False, feat_type=feat_types_cnn[i], max_length=MAX_LENGTH
    )
    
    backbone = CNNBackbone(input_shape_cnn, cnn_in_channels, cnn_filters, cnn_out_feature_size)
    model = Classifier(NUM_CATEGORIES, backbone) 
    
    model.to(DEVICE)
    print("For {} model:".format(save_paths_cnn[i]))
    print(f"Accuracy: {test(model, test_dataset, os.path.join(OUTPUT_DATA_DIR,save_paths_cnn[i]), BATCH_SIZE)}",'\n')
    
    del model    
    gc.collect()

"""**Step 8**"""

class Regressor(nn.Module):
    def __init__(self, backbone, load_from_checkpoint=None):
        """
        backbone (nn.Module): The nn.Module to use for spectrogram parsing
        load_from_checkpoint (Optional[str]): Use a pretrained checkpoint to initialize the model
        """
        super(Regressor, self).__init__()
        self.backbone = backbone  # An LSTMBackbone or CNNBackbone
        if load_from_checkpoint is not None:
            self.backbone = load_backbone_from_checkpoint(
                self.backbone, load_from_checkpoint
            )
        self.is_lstm = isinstance(self.backbone, LSTMBackbone)
        self.output_layer = nn.Linear(self.backbone.feature_size, 1)
        self.criterion = nn.MSELoss()  # Loss function for regression

    def forward(self, x, targets, lengths):
        feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)
        out = self.output_layer(feats)
        loss = self.criterion(out.float(), targets.float().unsqueeze(1))
        return loss, out

# Custom function for train and validation split for current datasets
def torch_train_val_test_split(
    dataset, batch_train, batch_eval,batch_test, val_size=0.2,test_size=0.1, shuffle=True, seed=420
):
    # Creating data indices for testing validation and training splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    test_split = int(np.floor(test_size * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    rest_indices = indices[test_split:]
    test_indices = indices[:test_split]
    
    val_split = int(np.floor(val_size * len(rest_indices)))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(rest_indices)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_train, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_eval, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_test, sampler=test_sampler)
    return train_loader, val_loader, test_loader

# Overfitting LSTM model for regression
train_dataset = SpectrogramDataset(
        os.path.join(PARENT_DATA_DIR, 'multitask_dataset_beat'), train=True,  regression=1, feat_type='',max_length = MAX_LENGTH)

train_loader, val_loader = torch_train_val_split(train_dataset, BATCH_SIZE, BATCH_SIZE)

backbone = LSTMBackbone(input_shape[4], rnn_size=RNN_HIDDEN_SIZE[1], num_layers=2, bidirectional=True)
model = Regressor(backbone) 

model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train(model, train_loader, val_loader, optimizer, epochs[0], device=DEVICE, overfit_batch=True)

reg_lstm_save_paths = ['reg_1_lstm_genre_all_beat.pth','reg_2_lstm_genre_all_beat.pth','reg_3_lstm_genre_all_beat.pth']
reg_cnn_save_paths = ['reg_1_cnn_genre_mel_beat.pth','reg_2_cnn_genre_mel_beat.pth','reg_3_cnn_genre_mel_beat.pth']

test_loader_lstm = [0,0,0]

for i in range(0,3):
    dataset = SpectrogramDataset(
            os.path.join(PARENT_DATA_DIR, 'multitask_dataset_beat'), train=True,  regression=i+1, feat_type='',max_length = MAX_LENGTH)

    train_loader, val_loader, test_loader_lstm[i] = torch_train_val_test_split(dataset, BATCH_SIZE,BATCH_SIZE, BATCH_SIZE)

    backbone = LSTMBackbone(input_shape[4], rnn_size=RNN_HIDDEN_SIZE[1], num_layers=2, bidirectional=True)
    model = Regressor(backbone) 

    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Overfitting
    # train(model, train_loader, val_loader, optimizer, epochs[0],  device=DEVICE, overfit_batch=True)
    train(model, train_loader, val_loader, optimizer, epochs[1], save_path=reg_lstm_save_paths[i], device=DEVICE, overfit_batch=False) 
    
    # Release memory from RAM
    del model
    del train_loader
    del val_loader
    del dataset

    # Call garbage collector
    gc.collect()

test_loader_CNN = [0,0,0]

for i in range(0,3):
    dataset = SpectrogramDataset(
            os.path.join(PARENT_DATA_DIR, 'multitask_dataset_beat'), train=True,  regression=i+1, feat_type='mel',max_length = MAX_LENGTH)

    train_loader, val_loader, test_loader_CNN[i] = torch_train_val_test_split(dataset, BATCH_SIZE,BATCH_SIZE, BATCH_SIZE)

    backbone = CNNBackbone(input_shape_cnn, cnn_in_channels, cnn_filters, cnn_out_feature_size)
    model = Regressor(backbone) 
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Overfitting
    # train(model, train_loader, val_loader, optimizer, epochs[0],  device=DEVICE, overfit_batch=True)
    train(model, train_loader, val_loader, optimizer, epochs[1],  device=DEVICE, save_path=reg_cnn_save_paths[i], overfit_batch=False) 
    
    # Release memory from RAM
    del model
    del train_loader
    del val_loader
    del dataset

    # Call garbage collector
    gc.collect()

from torchmetrics import SpearmanCorrCoef

# Write test function for regression
def regression_test(model, test_loader, path, batch_size, regression=1, device=DEVICE):   
       
    model.load_state_dict(torch.load(path))
    model.eval()
    
    predicted = torch.Tensor([])
    ground_truth = torch.Tensor([])
    
    # Find true and predicted labels
    with torch.no_grad():
        for x, y, lengths in test_loader:
            _, logits = model(x.float().to(device), y.to(device), lengths.to(device))
            predicted = torch.cat((predicted,logits.to('cpu')))
            ground_truth = torch.cat((ground_truth,y))
    predicted = torch.squeeze(predicted, 1)     
    
    # Calculate Spearman correlation
    spearman = SpearmanCorrCoef(num_outputs=regression)

    # Calculate spearman correlation   
    return spearman(predicted, ground_truth)

# Total testing for all regression LSTM models (for 3 different models)
for i,p in enumerate(reg_lstm_save_paths):    
    backbone = LSTMBackbone(input_shape[4], rnn_size=RNN_HIDDEN_SIZE[1], num_layers=2, bidirectional=True)
    model = Regressor(backbone) 
    
    model.to(DEVICE)
    print("For {} model:".format(p))
    print(f"Spearman Correlation: {regression_test(model, test_loader_lstm[i], os.path.join(OUTPUT_DATA_DIR,p), BATCH_SIZE)}",'\n')
    
    del model    
    gc.collect()

# Total CNN testing for regression models (for 3 different models)
for i,p in enumerate(reg_cnn_save_paths): 
    
    backbone = CNNBackbone(input_shape_cnn, cnn_in_channels, cnn_filters, cnn_out_feature_size)
    model = Regressor(backbone)
    
    model.to(DEVICE)
    print("For {} model:".format(p))
    print(f"Spearman Correlation: {regression_test(model, test_loader_CNN[i], os.path.join(OUTPUT_DATA_DIR,p), BATCH_SIZE)}",'\n')
    
    del model    
    gc.collect()

best_model_path = 'best_model.pth'
finetuned_model_path = 'finetuned_model.pth'

# Train best mondel
train_dataset = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, 'fma_genre_spectrograms_beat'), class_mapping=CLASS_MAPPING, 
    train=True, feat_type='', max_length=MAX_LENGTH
)

train_loader, val_loader = torch_train_val_split(train_dataset, BATCH_SIZE, BATCH_SIZE)

# Run training
backbone = CNNBackbone(input_shape_cnn, cnn_in_channels, cnn_filters, cnn_out_feature_size)
model = Classifier(NUM_CATEGORIES, backbone)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train(model, train_loader, val_loader, optimizer, epochs[1], device=DEVICE, overfit_batch=False)
print(backbone.state_dict())
torch.save(backbone.state_dict(), best_model_path)

def load_backbone_from_checkpoint(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))

# Fine-tune best model
dataset = SpectrogramDataset(
            os.path.join(PARENT_DATA_DIR, 'multitask_dataset_beat'), train=True,  regression=1, feat_type='',max_length = MAX_LENGTH)

train_loader, val_loader, test_loader = torch_train_val_test_split(dataset, BATCH_SIZE,BATCH_SIZE, BATCH_SIZE)

backbone = CNNBackbone(input_shape_cnn, cnn_in_channels, cnn_filters, cnn_out_feature_size)
load_backbone_from_checkpoint(backbone, checkpoint_path=best_model_path)
model = Regressor(backbone)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train(model, train_loader, val_loader, optimizer, epochs[1], save_path=finetuned_model_path, device=DEVICE, overfit_batch=False)

# Test fine-tuned model
backbone = CNNBackbone(input_shape_cnn, cnn_in_channels, cnn_filters, cnn_out_feature_size)
model = Regressor(backbone)

model.to(DEVICE)
print("For {} model:".format(finetuned_model_path))
print(f"Spearman Correlation: {regression_test(model, test_loader, os.path.join(OUTPUT_DATA_DIR,finetuned_model_path), BATCH_SIZE)}",'\n')