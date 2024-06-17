# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import nn, optim
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 384),
            nn.LeakyReLU(),

        )
        self.decoder = nn.Sequential(
            nn.Linear(384, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 1024),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
 

epochs = 50
test_sample_ratio = .2

input_dir = "/Volumes/Research/GWNMMAD_data/Downsampled/V1/"
input_file_list = ["H-H1_GWOSC_O2_4KHZ_R1-1174978560-4096_downsampled", "L-L1_GWOSC_O2_4KHZ_R1-1187291136-4096_downsampled", "V-V1_GWOSC_O2_4KHZ_R1-1187688448-4096_downsampled"]

data = np.reshape(pd.read_csv(input_dir + input_file_list[0] + ".csv", header=None)[0].to_numpy(), (-1, 1024))

assert data.shape[1] == 1024

total_sample_number = data.shape[0]
test_sample_number = int(test_sample_ratio * total_sample_number)


X_train = data[0:test_sample_number-1]
X_test = data[-test_sample_number:0]


trainData = torch.FloatTensor(X_train)
testData = torch.FloatTensor(X_test)


train_dataset = TensorDataset(trainData, trainData)
test_dataset = TensorDataset(testData, testData)
trainDataLoader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True)


autoencoder = AutoEncoder()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
loss_func = nn.MSELoss()
loss_train = np.zeros((epochs, 1))


for epoch in range(epochs):
    # 不需要label，所以用一个占位符"_"代替
    for batchidx, (x, _) in enumerate(trainDataLoader):
        # 编码和解码 
        encoded, decoded = autoencoder(x)
        # 计算loss
        loss = loss_func(decoded, x)      
        # 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
           
    loss_train[epoch,0] = loss.item()  
    print('Epoch: %04d, Training loss=%.8f' %
          (epoch+1, loss.item()))


# 绘制loss曲线
fig = plt.figure(figsize=(6, 3))
ax = plt.subplot(1, 1, 1)
ax.grid()
ax.plot(loss_train, color=[245/255, 124/255, 0/255], linestyle='-', linewidth=2)    
ax.set_xlabel('Epoches')
ax.set_ylabel('Loss')
plt.show()

# 利用训练好的自编码器重构测试数据
_, decodedTestdata = autoencoder(testData)
decodedTestdata = decodedTestdata.double()
reconstructedData = decodedTestdata.detach().numpy()


mse = ((reconstructedData - testData)).mean()
print(mse)