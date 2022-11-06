import numpy as np
import time
import fruit_main as fm
import math
import matplotlib.pyplot as plt

cost_plotting_list = []


holy = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # one hot encoding 준비
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]


class LogisticRegression:
    x = fm.train_x  # x_train
    y = fm.train_label  # t_train
    m = 0  # data set 갯수
    n = 0  # input feature 갯수
    # np.random.randn(30000, 33)  # 초기 weight 값 랜덤 생성 + bias 추가
    b = np.array([])  # bias
    lr = 0  # learning rate
    epoch = 0  # epoch (학습횟수)

    def __init__(self, m, n, lr, epoch):  # 초기화
        self.lr = lr
        self.m = m
        self.n = n
        self.epoch = epoch
        self.w = np.random.randn(30000, 33)  # np.array([[0.7]*990000])
        #self.w = self.w.reshape(30000, 33)

    def learn(self):
        for i in range(0, self.epoch):
            cost1 = self.cost(self.x, self.y)  # cost 값
            cost_plotting_list.append(cost1)  # plotting 위한 배열 생성
            print("epoch:", i, "cost: ", cost1)  # cost 값 출력

            grad = self.gradient_descent(self.x, self.y)  # 기울기 산출

            self.w -= self.lr * grad  # weight 값 조정
        return

    def cost(self, x, t):
        h = self.predict(x)  # h() 가설함수

        real_cost = np.sum((t*np.log(h + 0.0000001) + (1 - t)  # log안 변수 0방지 위한 수 더해줌.
                             * np.log(1 - h + 0.0000001)), axis=0)/self.m

        return -real_cost

    def predict(self, x):  # 예측함수
        w = self.w  # weight 행렬

        y = np.dot(x, w)  # W * x
        h = self.sigmoid(y)  # sigmoid( W * x) -> h() 가설함수

        return h

    def sigmoid(self, x):  # sigmoid 함수
        return 1/(1 + np.exp(-x))

    def gradient_descent(self, x, t):  # 경사 하강 알고리즘

        h = self.predict(x)  # 가설함수

        grad = np.dot(x.T, h - t)  # 편미분 값
        # print(grad)
        return grad


target = LogisticRegression(100, 30000, 0.00001, 500)
target.learn()  # 학습 시작


a = np.arange(0, 500, 1)
b = cost_plotting_list
plt.plot(a, b)
plt.show()
