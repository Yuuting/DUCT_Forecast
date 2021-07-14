# -*- coding: utf-8 -*-
# @Time    : 2021/7/12 21:21
# @Author  : yuting
# @FileName: timeSeries.py
import numpy as np
import pandas as pd
from loguru import logger
from model.lstm import RegLSTM
import torch
import matplotlib.pyplot as plt
from torch import nn

# 时序数据路径
path = "./时序数据/nx=1ny=1nl=1.csv"


class dataPreprocess:
    def load_data(path):
        raw_seq = pd.read_csv(path)
        print("三个月的数据共有" + str(raw_seq.shape[0]) + "条")
        '''
        plt.plot(raw_seq["Tem"])
        plt.ion()
        plt.pause(1)
        '''
        seq = raw_seq[['nt', 'e', 'pressure', 'Tem', 'QVAPOR']]
        # normolization
        seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)
        return np.array(seq)


class train_data:
    def run_train_lstm(seq):
        # lstm模型参数
        inp_dim = 5
        out_dim = 4
        mid_dim = 8
        mid_layers = 1
        batch_size = 48
        mod_dir = '.'

        data_x = seq[:-1, :]
        data_y = seq[1:, 1:]
        print(data_x.shape)
        print(data_y.shape)
        assert data_x.shape[1] == inp_dim

        train_size = int(len(data_x) * 0.75)

        train_x = data_x[:train_size]
        train_y = data_y[:train_size]
        train_x = train_x.reshape((train_size, inp_dim))
        train_y = train_y.reshape((train_size, out_dim))

        '''build model'''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = RegLSTM(inp_dim, out_dim, mid_dim, mid_layers).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

        '''train'''
        var_x = torch.tensor(train_x, dtype=torch.float32, device=device)
        var_y = torch.tensor(train_y, dtype=torch.float32, device=device)

        batch_var_x = list()
        batch_var_y = list()

        for i in range(batch_size):
            j = train_size - i
            batch_var_x.append(var_x[j:])
            batch_var_y.append(var_y[j:])

        from torch.nn.utils.rnn import pad_sequence
        batch_var_x = pad_sequence(batch_var_x)
        batch_var_y = pad_sequence(batch_var_y)

        with torch.no_grad():
            weights = np.tanh(np.arange(len(train_y)) * (np.e / len(train_y)))
            weights = torch.tensor(weights, dtype=torch.float32, device=device)

        print("Training Start")
        for e in range(384):
            out = net(batch_var_x)

            loss = criterion(out, batch_var_y)
            # loss = (out - batch_var_y) ** 2 * weights
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if e % 64 == 0:
                print('Epoch: {:4}, Loss: {:.5f}'.format(e, loss.item()))
        torch.save(net.state_dict(), '{}/net.pth'.format(mod_dir))
        print("Save in:", '{}/net.pth'.format(mod_dir))

        '''eval'''
        net.load_state_dict(torch.load('{}/net.pth'.format(mod_dir), map_location=lambda storage, loc: storage))
        net = net.eval()

        test_x = data_x.copy()
        test_x[train_size:, 1:5] = 0
        test_x = test_x[:, np.newaxis, :]
        test_x = torch.tensor(test_x, dtype=torch.float32, device=device)

        eval_size = 1
        zero_ten = torch.zeros((mid_layers, eval_size, mid_dim), dtype=torch.float32, device=device)
        test_y, hc = net.output_y_hc(test_x[:train_size], (zero_ten, zero_ten))
        test_x[train_size + 1, 0, 1:5] = test_y[-1]
        for i in range(train_size + 1, len(seq) - 2):
            test_y, hc = net.output_y_hc(test_x[i:i + 1], hc)
            test_x[i + 1, 0, 1:5] = test_y[-1]

        pred_y = test_x[1:, 0, 1:5]
        pred_y = pred_y.cpu().data.numpy()

        diff_y = pred_y[train_size:] - data_y[train_size:-1]
        l1_loss = np.mean(np.abs(diff_y))
        l2_loss = np.mean(diff_y ** 2)
        print("L1: {:.3f}    L2: {:.3f}".format(l1_loss, l2_loss))

        plt.plot(pred_y[:,1], 'r', label='pred')
        plt.plot(data_y[:,1], 'b', label='real')
        # plt.plot([train_size, train_size], [-1, 2], color='k', label='train | pred')
        plt.legend(loc='best')
        plt.savefig('lstm_reg.png')
        plt.pause(30)


if __name__ == '__main__':
    logger.info("开始数据预处理过程")
    d = dataPreprocess
    seq = d.load_data(path)

    logger.info("开始进行lstm模型训练")
    tr = train_data
    tr.run_train_lstm(seq)
