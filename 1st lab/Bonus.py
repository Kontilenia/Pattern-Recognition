import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

absolute_path = os.path.dirname(__file__)
data_path = ["test.txt", "train.txt"]
data_folder = "data"
full_path_test = os.path.join(absolute_path, data_folder, data_path[0])
full_path_train = os.path.join(absolute_path, data_folder, data_path[1])

test = pd.read_csv(full_path_test, sep=' ', header=None, index_col=False)
test.dropna(axis=1, how='all', inplace=True)

train = pd.read_csv(full_path_train, sep=' ', header=None, index_col=False)
train.dropna(axis=1, how='all', inplace=True)


class ImageData(Dataset):
    def __init__(self, train_data):
        # all the available data are stored the class
        self.data = train_data.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :]


class NonLinearActivation(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super(NonLinearActivation, self).__init__()
        self.f = nn.Linear(in_features, out_features)
        self.a = activation

    def forward(self, x):
        output = self.f(x)
        return self.a(output)


class MyNNetwork(nn.Module):
    def __init__(self, layers, n_features, n_classes, activation):
        super(MyNNetwork, self).__init__()
        layers_in = [n_features] + layers
        layers_out = layers + [n_classes]
        self.f = nn.Sequential(*[
            NonLinearActivation(in_feats, out_feats, activation=activation)
            for in_feats, out_feats in zip(layers_in, layers_out)
        ])

    def forward(self, x):
        y = self.f(x)
        return y


EPOCHS = 20
# the mini-batch size
BATCH_SZ = 64
learning_rate = 1e-5

# Test different non linear activate functions
net = MyNNetwork([500, 20], train.shape[1] - 1, 10, nn.Sigmoid())
# net = MyNNetwork([50, 100, 20], train.shape[1] - 1, 10, nn.ReLU())
# net = MyNNetwork([80, 400], train.shape[1]-1, 10, nn.Tanh())
print(f"The network architecture is: \n {net}")

# define the loss function
criterion = nn.CrossEntropyLoss()

# define the optimizer

optimizer = optim.SGD(net.parameters(), lr=learning_rate)

train_set = ImageData(train)
test_set = ImageData(test)

train_dl = DataLoader(train_set, batch_size=BATCH_SZ, shuffle=True)
test_dl = DataLoader(test_set, batch_size=BATCH_SZ, shuffle=True)

# gradients on
net.train()
# for each epoch
for epoch in range(EPOCHS):
    running_average_loss = 0
    # for every batch
    for i, data in enumerate(train_dl):
        all = data
        # get the features and labels
        X_batch, y_batch = all[:, 1:], all[:, 0]
        optimizer.zero_grad()
        out = net(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        # update weights
        optimizer.step()

        running_average_loss += loss.detach().item()
        if i % 100 == 0:
            print("Epoch: {} \t Batch: {} \t Loss {}".format(epoch, i, float(running_average_loss) / (i + 1)))

# turns off dropout for testing
net.eval()
acc = 0
n_samples = 0
with torch.no_grad():
    for i, data in enumerate(test_dl):
        all = data
        # test data and labels
        X_batch, y_batch = all[:, 1:], all[:, 0]
        # get net's predictions
        out = net(X_batch)
        val, y_pred = out.max(1)
        # get accuracy
        acc += (y_batch == y_pred).sum().detach().item()
        n_samples += X_batch.size(0)

# Print accuracy
print(acc)
print(n_samples)
print(acc / n_samples)
