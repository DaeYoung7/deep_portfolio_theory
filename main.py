import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

class MyModel(nn.Module):
    def __init__(self, layer, num_feature, encoded_size):
        super().__init__()
        self.layer = layer
        self.input_shape = num_feature
        self.encoded_size = encoded_size
        self.hidden_size = 128
        self.output_shape = num_feature

        self.encoder_hl1 = nn.Linear(self.input_shape, self.hidden_size)
        self.encoder_norm = nn.BatchNorm1d(self.hidden_size)
        self.encoder_hl2 = nn.Linear(self.hidden_size, self.encoded_size)
        self.encoder_lstm = nn.LSTM(self.input_shape, self.encoded_size, 1)
        self.decoder = nn.Linear(self.encoded_size, self.output_shape)
    def forward(self, inputs):
        if self.layer == 'linear':
            x = torch.relu(self.encoder_hl1(inputs))
            x = self.encoder_hl2(self.encoder_norm(x))
        else:
            x, _ = self.encoder_lstm(inputs)
            x = torch.squeeze(x[:,-1,:], dim=1)
        output = self.decoder(torch.relu(x))
        return output, x

data = pd.read_csv('K200_stationary.csv', index_col='date', parse_dates=True)
split = int((len(data)-252) * 0.8)
seq_data = []
y = []
i = 252
layer_type = 'linear'
if layer_type == 'lstm':
    while i < len(data):
        seq_data.append(data[i-252:i].values)
        y.append(data.iloc[i-1].values)
        i += 1
    train_data = torch.tensor(np.array(seq_data[:split], dtype=np.float32))
    train_label = torch.tensor(np.array(y[:split], dtype=np.float32))
    test_data = torch.tensor(np.array(seq_data[split:], dtype=np.float32))
    test_label = torch.tensor(np.array(y[split:], dtype=np.float32))
else:
    train_data = torch.tensor(data[:split].values, dtype=torch.float32)
    train_label = torch.tensor(data[:split].values, dtype=torch.float32)
    test_data = torch.tensor(data[split:].values, dtype=torch.float32)
    test_label = torch.tensor(data[split:].values, dtype=torch.float32)

num_feature = len(data.columns)
encoded_size = num_feature // 4
encoded_size += (4 - encoded_size % 4)
print(f'encoded_size : {encoded_size}')
net = MyModel(layer_type, num_feature, encoded_size)
optimizer = Adam(net.parameters(), lr=0.01, weight_decay=1e-5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)
loss_fn = nn.MSELoss()

epochs = 15000
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
    scheduler.step()
    if (epoch+1) % 10 == 0:
        print(f'{epoch+1}  {tloss}  {vloss}')
plt.plot(tlosses, label='train')
plt.plot(vlosses, label='test')
plt.legend()
plt.show()
time.sleep(3)
plt.clf()

total_data = torch.tensor(data.values, dtype=torch.float32)
output, encoded_data = net(total_data)
output = output.detach().numpy()

for i in range(len(data.columns)):
    plt.plot(data[data.columns[i]][split:], label='original')
    plt.plot(data.index[split:], output[split:,i], label='simul')
    plt.legend()
    plt.show()

output_df = pd.DataFrame(encoded_data)
output_df.index = data.index
output_df.to_csv('K200_deep_factor.csv')