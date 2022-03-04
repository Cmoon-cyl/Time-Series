#!/usr/bin/env python
# coding: UTF-8
# Created by Cmoon

import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader


class Data:
    def __init__(self, path):
        self.path = path
        self.raw_data = pd.read_csv(self.path, index_col=0)
        self.mean = self.raw_data.mean()
        self.std = self.raw_data.std()
        self.step = 60
        self.train_set = []
        self.test_set = []
        self.train_features = []
        self.train_labels = []
        self.test_features = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def preprocess(self):
        self.normalization()
        self.divide()
        return self.train_features, self.train_labels, self.test_features

    def normalization(self):
        numeric_features = self.raw_data.dtypes[self.raw_data.dtypes != 'object'].index  # 非数据值
        self.raw_data[numeric_features] = self.raw_data[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std()))
        self.raw_data[numeric_features] = self.raw_data[numeric_features].fillna(0)  # 缺失值置0

    def divide(self):
        n_train = int(0.7 * len(self.raw_data))
        self.train_set = self.raw_data[:n_train].values
        self.test_set = self.raw_data[n_train:].values
        for i in range(self.step, n_train):
            self.train_features.append(self.train_set[i - self.step:i, 0])
            self.train_labels.append(self.train_set[i, 0])
        for i in range(self.step, len(self.raw_data) - n_train):
            self.test_features.append(self.test_set[i - self.step:i, 0])
        self.train_features = torch.tensor(np.array(self.train_features), dtype=torch.float32).to(self.device)
        self.train_labels = torch.tensor(np.array(self.train_labels), dtype=torch.float32).view(-1, 1).to(self.device)
        self.test_features = torch.tensor(np.array(self.test_features), dtype=torch.float32).to(self.device)

    @staticmethod
    def load_array(data_arrays, batch_size, is_train=True):
        dataset = TensorDataset(*data_arrays)
        return DataLoader(dataset, batch_size, shuffle=is_train)


class Model:
    def __init__(self, dataset, num_epochs=500, batch_size=64,
                 learning_rate=1e-3, weight_decay=0.0, k=5):
        self.in_features = dataset.train_features.shape[1]
        self.loss = nn.MSELoss()
        self.dataset = dataset
        self.train_features = dataset.train_features
        self.train_labels = dataset.train_labels
        self.epochs = num_epochs
        self.bs = batch_size
        self.lr = learning_rate
        self.wd = weight_decay
        self.k = k
        self.pre = []
        self.truth = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = nn.Sequential(
            nn.Linear(self.in_features, 60),
            nn.ReLU(),
            nn.Linear(60, 10),
            nn.ReLU(),
            nn.Linear(10, 1)).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=learning_rate,
                                          weight_decay=weight_decay)

    def train(self, train_features, train_labels, test_features, test_labels):
        train_ls, test_ls = [], []
        train_iter = Data.load_array((train_features, train_labels), self.bs)
        # 这里使用的是Adam优化算法
        for epoch in range(self.epochs):
            for X, y in train_iter:
                self.optimizer.zero_grad()
                l = self.loss(self.net(X), y)
                l.backward()
                self.optimizer.step()
            train_ls.append(l)
            if test_labels is not None:
                test_ls.append(self.loss(self.net(test_features), test_labels))
        return train_ls, test_ls

    def get_k_fold_data(self, i, X, y):
        assert self.k > 1
        fold_size = X.shape[0] // self.k
        X_train, y_train, X_valid, y_valid = None, None, None, None
        for j in range(self.k):
            idx = slice(j * fold_size, (j + 1) * fold_size)
            X_part, y_part = X[idx, :], y[idx]
            if j == i:
                X_valid, y_valid = X_part, y_part
            elif X_train is None:
                X_train, y_train = X_part, y_part
            else:
                X_train = torch.cat([X_train, X_part], 0)
                y_train = torch.cat([y_train, y_part], 0)
        return X_train, y_train, X_valid, y_valid

    def k_fold(self):
        train_l_sum, valid_l_sum = 0, 0
        for i in range(self.k):
            data = self.get_k_fold_data(i, self.train_features, self.train_labels)
            train_ls, valid_ls = self.train(*data)
            train_l_sum += train_ls[-1]
            valid_l_sum += valid_ls[-1]
            print(f'fold {i + 1}, train mse {float(train_ls[-1]):f}, '
                  f'valid mse {float(valid_ls[-1]):f}')
        return train_l_sum / self.k, valid_l_sum / self.k

    def predict(self):
        predict = []
        for x in self.dataset.test_features:
            predict.append(self.net(x).detach().to('cpu').numpy())
        self.pre = np.array(list(map(lambda x: (x * self.dataset.std) + self.dataset.mean, predict)))
        self.truth = np.array(
            list(map(lambda x: (x * self.dataset.std) + self.dataset.mean, self.dataset.test_set[60:])))

    def plot(self):
        plt.plot(self.truth, color='red', label='Ground truth')
        plt.plot(self.pre, color='blue', label='Prediction')
        plt.title('wind speed')
        plt.xlabel('time')
        plt.ylabel('speed')
        plt.legend()
        plt.show()

    def eval(self):
        self.predict()
        self.plot()


if __name__ == '__main__':
    data = Data(r'shww1.csv')
    data.preprocess()
    trainer = Model(data, num_epochs=500, batch_size=64,
                    learning_rate=1e-3, weight_decay=1e-4, k=5)
    train_l, valid_l = trainer.k_fold()
    print(f'{trainer.k}-折验证: 平均训练mse: {train_l}, '
          f'平均验证mse: {valid_l}')
    trainer.eval()
