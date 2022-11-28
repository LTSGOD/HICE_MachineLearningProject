import numpy as np
from collections import OrderedDict


class Affine:
    def __init__(self, W, b):
        self.W, self.b = W, b
        self.x, self.dW, self.db = None, None, None

    def forward(self, x, traing_flg=True):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x, train_flg=True):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y, self.t = None, None

    def forward(self, x, t, train_flg=True):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):

        batch_size = self.t.shape[0]  # 100
        dx = (self.y - self.t)/batch_size

        return dx


def softmax(x):

    c = np.max(x, axis=1)

    exp_a = np.exp(x.T - c)
    sum_exp_a = np.sum(exp_a, axis=0)
    result = exp_a / sum_exp_a

    # print(result)
    return result.T

# soft max에서 문제발견 각행에서 최솟값을 빼야하는데 그냥 전체맥스값에서 빼버림


class Tanh:
    def __init__(self):
        self.mask = None

    def forward(self, x, train_flg=True):
        out = np.tanh(x)
        self.out = out

        return out

    def backward(self, dout):
        dx = dout*(1 - self.out*self.out)

        return dx


class LeakyReLU:
    def __init__(self):
        self.out = None

    def forward(self, z, train_flg=True):
        self.out = z
        self.out[self.out <= 0] *= 0.001
        return self.out

    def backward(self, dout):
        self.out[self.out > 0] = 1
        self.out[self.out <= 0] = 0.001
        return self.out * dout


class Dropout:
    def __init__(self, dropout_ratio=0.1):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

    result = -np.sum(t * np.log(y + 1e-7)) / batch_size
    # print(result)
    # -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    return result


def _numerical_gradient_no_batch_(f, x):

    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


def numerical_gradient(f, X):
    global count
    if X.ndim == 1:

        return _numerical_gradient_no_batch_
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch_(f, x)
            print(count)
            count = count + 1

        return grad


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}

        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        #print("초기 weight", self.params['W1'])

        # 계층생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Dropout'] = Dropout()

        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x, train_flg=True):

        for layer in self.layers.values():
            x = layer.forward(x, train_flg)
        return x

    # x: 입력 데이터, t : 정답레이블
    def cost(self, x, t):
        y = self.predict(x)
        result = self.lastLayer.forward(y, t)
        # print(result) 스칼라 (크로스엔트로피오차값)
        return result

    def accuracy(self, x, t, train_flg=True):
        y = self.predict(x, train_flg)
        y = np.argmax(y, axis=1)  # argmax는 요소가 최댓값인 index들을 리스트로 나타냄
        if t.ndim != 1:  # ndim은 차원의 수를 나타내며 one-hot-encoding이 되어있는 경우 실행
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, t):  # 가중치의 기울기를 수치미분으로 구함
        def cost_W(W): return self.cost(x, t)
        # grads : 기울기 보관하는 딕셔너리 변수
        grads = {}
        print(self.params['W1'].shape)
        grads['W1'] = numerical_gradient(
            cost_W, self.params['W1'])  # grads['W1']은 1층의 가중치의 기울기
        grads['b1'] = numerical_gradient(
            cost_W, self.params['b1'])  # grads['B1']은 1층의 편향의 기울기
        grads['W2'] = numerical_gradient(cost_W, self.params['W2'])
        grads['b2'] = numerical_gradient(cost_W, self.params['b2'])

        return grads

    def gradient(self, x, t):  # 가중치의 기울기를 오차 역전파로 구함
        # 순전파
        self.cost(x, t)
        # 역전파
        dout = 1  # 맨 마지막 층이므로 다음 층에서 흘러들어오는 값이 없어서 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())

        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db

        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
