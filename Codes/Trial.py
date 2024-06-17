# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import nn, optim
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# 自编码器
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.LeakyReLU(),
            nn.Linear(12, 48),
            nn.LeakyReLU(),

        )
        self.decoder = nn.Sequential(
            nn.Linear(48, 12),
            nn.LeakyReLU(),
            nn.Linear(12, 3),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
 
# 参数设置
epochs = 50

# 读取数据
matfile = sio.loadmat('./data/data.mat')
X_train = matfile['trainData']
X_test = matfile['testData']

# 数据类型转换 
trainData = torch.FloatTensor(X_train)
testData = torch.FloatTensor(X_test)

# 构建张量数据集
train_dataset = TensorDataset(trainData, trainData)
test_dataset = TensorDataset(testData, testData)
trainDataLoader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True)

# 初始化
autoencoder = AutoEncoder()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
loss_func = nn.MSELoss()
loss_train = np.zeros((epochs, 1))

# 训练
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

# 利用训练好的自编码器重构测试数据
_, decodedTestdata = autoencoder(testData)
decodedTestdata = decodedTestdata.double()
reconstructedData = decodedTestdata.detach().numpy()
