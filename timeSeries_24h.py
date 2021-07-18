# -*- coding: utf-8 -*-
# @Time    : 2021/7/15 22:39
# @Author  : yuting
# @FileName: timeSeries_24h.py

import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model.NN import Net
import pandas as pd
import numpy as  np

# 时序数据路径
path = "./时序数据/nx=1ny=1nl=1_wrfMode.csv"


def dataProcessing(path):
    allData = pd.read_csv(path)
    tem = allData['Tem'].copy()
    temData = []
    for i in range(2 * 31 + 27):
        temData.append(np.array(tem)[i * 25:i * 25 + 12])  # np.array(tem)[0:tem.shape[0]:25].reshape(-1, 1)
    temLabel = []
    for i in range(2 * 31 + 27):
        temLabel.append(np.array(tem)[i * 25 + 12:i * 25 + 25])
    x_train, x_test, y_train, y_test = train_test_split(temData, temLabel, test_size=0.25,
                                                        random_state=42)

    # 归一化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    Y_train = scaler.fit_transform(y_train)
    X_test = scaler.fit_transform(x_test)
    Y_test = scaler.fit_transform(y_test)

    return X_train, Y_train, X_test, Y_test


def train_data(x_train, y_train, x_test, y_test):
    X_train = torch.from_numpy(x_train.astype(np.float32))
    Y_train = torch.from_numpy(y_train.astype(np.float32))
    X_test = torch.from_numpy(x_test.astype(np.float32))
    Y_test = torch.from_numpy(y_test.astype(np.float32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    print("Training Start")
    for e in range(3500):
        out = net(X_train)

        loss = criterion(out, Y_train)
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 64 == 0:
            print('Epoch: {:4}, Loss: {:.5f}'.format(e, loss.item()))

    net = net.eval()
    pred_y = net(X_test)
    pred_y = pred_y.cpu().data.numpy()
    diff_y = pred_y - y_test

    l1_loss = np.mean(np.abs(diff_y))
    l2_loss = np.mean(diff_y ** 2)
    print("L1: {:.3f}    L2: {:.3f}".format(l1_loss, l2_loss))

    plt.figure(dpi=250)
    plt.plot(pred_y.flatten(), 'r', label='pred')
    plt.plot(y_test.flatten(), 'b', label='real')
    # 设置刻度的字号
    plt.xlabel('Hours/h', fontsize=14)
    # 设置x轴标签及其字号
    plt.ylabel('Temperature/K', fontsize=14)
    plt.title('Temperature chart of Nanhai (1, 1, 1) from January to March', fontsize='large', fontweight='bold')
    plt.legend(loc='best')
    plt.savefig('NN_12-13h.png')
    plt.pause(10)


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = dataProcessing(path)
    train_data(X_train, Y_train, X_test, Y_test)
