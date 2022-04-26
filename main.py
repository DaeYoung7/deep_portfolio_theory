import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam

class MyModel(nn.Module):
    def __init__(self, num_feature, encoded_size):
        super().__init__()
        self.input_shape = num_feature
        self.encoded_size = encoded_size
        self.hidden_size = 256
        self.output_shape = num_feature

        # self.encoder_hl1 = nn.Linear(self.input_shape, self.hidden_size)
        # self.encoder_hl2 = nn.Linear(self.hidden_size, self.encoded_size)
        self.encoder_hl1 = nn.LSTM(self.input_shape, self.encoded_size, 1)
        self.decoder = nn.Linear(self.encoded_size, self.output_shape)
    def forward(self, inputs):
        x, _ = self.encoder_hl1(inputs)
        x = torch.squeeze(x[:,-1,:], dim=1)
        # x = torch.relu(self.encoder_hl2(x))
        output = self.decoder(torch.relu(x))
        return output, x

data = pd.read_csv('K200_stationary.csv', index_col='date', parse_dates=True)
split = int((len(data)-252) * 0.8)
seq_data = []
y = []
i = 252
while i < len(data):
    seq_data.append(data[i-252:i].values)
    y.append(data.iloc[i-1].values)
    i += 1
# train_data = torch.tensor(data[:split].values, dtype=torch.float32)
# test_data = torch.tensor(data[split:].values, dtype=torch.float32)

train_data = torch.tensor(np.array(seq_data[:split], dtype=np.float32))
train_label = torch.tensor(np.array(y[:split], dtype=np.float32))
test_data = torch.tensor(np.array(seq_data[split:], dtype=np.float32))
test_label = torch.tensor(np.array(y[split:], dtype=np.float32))

num_feature = len(data.columns)
encoded_size = num_feature // 10 + 1
net = MyModel(num_feature, encoded_size)
optimizer = Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
loss_fn = nn.MSELoss()

epochs = 5000
tlosses = []
vlosses = []
for epoch in range(epochs):
    net.train()
    optimizer.zero_grad()
    toutput, encoded_train_data = net(train_data)
    tloss = loss_fn(train_label, toutput)
    tloss.backward()
    optimizer.step()

    net.eval()
    voutput, encoded_test_data = net(test_data)
    vloss = loss_fn(test_label, voutput)

    tlosses.append(tloss)
    vlosses.append(vloss)

    print(f'{epoch}  {tloss}  {vloss}')
plt.plot(tlosses, label='train')
plt.plot(vlosses, label='test')
plt.legend()
plt.show()
time.sleep(3)
plt.clf()

total_data = torch.tensor(data.values, dtype=torch.float32)
output, encoded_data = net(total_data)
output = output.detach().numpy()

score = np.linalg.norm(output - data.values, axis=0)
score = pd.Series(score, index=data.columns).sort_values()
# best_idx, worst_idx = list(data.columns).index(score.index[0]), list(data.columns).index(score.index[-1])
# plt.plot(data[score.index[0]], label='best_original')
# plt.plot(data.index, output[:,best_idx], label='best_simul')
# plt.legend()
# plt.show()
#
# plt.plot(data[score.index[-1]], label='worst_original')
# plt.plot(data.index, output[:,worst_idx], label='worst_simul')
# plt.show()
for i in range(len(data.columns)):
    plt.plot(data[data.columns[i]][split:], label='original')
    plt.plot(data.index[split:], output[split:,i], label='simul')
    plt.legend()
    plt.show()