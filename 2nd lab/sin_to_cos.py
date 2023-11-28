import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.autograd import Variable

f = 40.
w = 2. * np.pi
time_interval = np.arange(0, 1000 / f, 0.002)
sin_all = np.sin(w * time_interval * f)
cos_all = np.cos(w * time_interval * f)
a = np.random.randint(0, len(sin_all) - 10, 1000, dtype=int)
X = np.zeros((1000, 10))
y = np.zeros((1000, 10))
for i in range(len(a)):
    X[i, :] = sin_all[a[i]:a[i] + 10]
    y[i, :] = cos_all[a[i]:a[i] + 10]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=2000,
                                                    shuffle=False)


class SinDataset(Dataset):
    def __init__(self, X, y):
        # all the available data are stored in a list
        self.data = np.zeros((X.shape[0], 2 * X.shape[1]))
        self.data[:, 0:X.shape[1]] = X
        self.data[:, X.shape[1]:] = y
        self.data = torch.tensor(self.data).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


train_dataset = SinDataset(X_train, y_train)
test_dataset = SinDataset(X_test, y_test)

torch.manual_seed(101)
batch_size = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class RNNModel(nn.Module):
    def __init__(self, input_dim,seq_len, hidden_dim, num_lay, out_dim):
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.num_layer = num_lay
        self.seq_len = seq_len

        # RNN layers
        self.rnn = nn.RNN(
            input_dim, hidden_dim, num_lay,batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim , out_dim*seq_len)

    def forward(self, x):
        # Add one more dimension
        x = x.unsqueeze(-1)

        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.num_layer, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length * hidden_dim)
        # out = out.contiguous().view(batch_size, -1)
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim,cell_dim, layer_dim, out_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions and hidden cells
        self.hidden_dim = hidden_dim
        self.cell_dim = cell_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Number of sequence length
        self.seq_len = seq_len

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, out_dim*seq_len)

    def forward(self, x):
        # Add one more dimension
        x = x.unsqueeze(-1)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.cell_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# batch_size, epoch and iteration
n_iters = 8000
num_epochs = 100

# Create RNN
input_size = 1     # input dimension
seq_length = 10   # input dimension
hidden_size = 64  # hidden layer dimension
num_layer = 3     # number of hidden layers
output_dim = 1   # output dimension

# model = RNNModel(input_size,seq_length, hidden_size, num_layer, output_dim)
model = LSTMModel(input_size,seq_length, hidden_size, hidden_size, num_layer, output_dim)

# Cross Entropy Loss
error = nn.MSELoss()

# SGD Optimizer
learning_rate = 0.03
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_list = []
iteration_list = []
accuracy_list = []
count = 0

for epoch in range(num_epochs):
    for i in train_loader:
        sin_var = i[:,0:10]
        cos_val = i[:, 10:]
        train = sin_var
        true_val = cos_val

        # Clear gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(train)

        # Calculate softmax and ross entropy loss
        loss = error(outputs, true_val)

        # Calculating gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        count += 1

time_interval = np.arange(100/f + 1.0025, 104/f +1, 0.0025)
sin_all_test = np.sin(w * time_interval * f)
plt.scatter(w*time_interval,sin_all_test)
plt.show()

cos_all_test = np.cos(w * time_interval * f)
plt.scatter(w*time_interval,cos_all_test)
plt.show()

sin_all_test = sin_all_test.reshape((4, 10))
cos_all_test = cos_all_test.reshape((4,10))

test_dataset = SinDataset(sin_all_test,cos_all_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

outputs = []
for j in test_loader:
    val_sin_var = j[:, 0:10]
    val_cos_val = j[:, 10:]
    val_true_val = val_cos_val
    model.eval()
    # Forward propagation
    outputs = model(val_sin_var)

a = outputs.detach().numpy()
b = w*time_interval
plt.scatter(b,a)
plt.show()