import numpy as np


class Perceptron(object):

    # eta : 学習率
    # n_iter : トレーニング回数
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        :param X: トレーニングデータ（配列）
        :param y: 目的変数（配列）
        :return: self
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        # トレーニングデータ回数分トレーニグデータを反復
        for _ in range(self.n_iter):
            errors = 0
            # 各サンプルで重みを更新
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """ 総入力を計算 """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """ 1ステップ後のクラスラベルを返す """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
