import numpy as np
import warnings
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
from fruitNN import TwoLayerNet
import os
import time

fruit_name = ["Apple Braeburn", "Apple Granny", "Apricot", "Avocado", "Banana", "Blueberry", "Cactus fruit", "Cantaloupe", "Cherry", "Clementine", "Corn", "Cucumber Ripe", "Grape blue", "Kiwi", "Lemon", "Limes",
              "Mango", "Onion White", "Orange", "Papaya", "Passion Fruit", "Peach", "Pear", "Pepper green", "Pepper red", "Pineapple", "Plum", "Pomegranate", "Potato Red", "Raspberry", "Strawberry", "Tomato", "Watermelon"]

warnings.simplefilter(action='ignore', category=FutureWarning)

"""---------------------------------Image option setting--------------------------------------"""

# trainig data 파일경로
training_dir = "C:\\Users\\82104\\OneDrive\\HICEMachingLearningProject\\archive\\train\\train"
# test data 파일경로
test_dir = "C:\\Users\\82104\\OneDrive\\HICEMachingLearningProject\\archive\\test"

training_datagen = ImageDataGenerator(rescale=1./255)  # 정규화

test_datagen = ImageDataGenerator(rescale=1./255)  # 정규화

batch_size = 100  # mini_batch 이용

training_generator = training_datagen.flow_from_directory(
    training_dir,
    # traing data는 100개를 매번 불러오는게 아닌 모든 data를 불러오고 mask를 통해 배치시킴(성능향상)
    batch_size=16858,
    target_size=(100, 100),  # target 크기 100 x 100
    class_mode='categorical',  # one hot encoding 사용
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    batch_size=batch_size,
    class_mode='categorical',  # one hot encoding 사용
    target_size=(100, 100),  # target 크기 100 x 100
)


class Adam:
    def __init__(self, lr=0.0001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        self.m_hat = None
        self.v_hat = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v, self.m_hat, self.v_hat = {}, {}, {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
                self.m_hat[key] = np.zeros_like(val)
                self.v_hat[key] = np.zeros_like(val)

        #self.iter += 1

        for key in ('W1', 'b1', 'W2', 'b2'):
            self.m[key] = self.beta1 * self.m[key] + \
                (1 - self.beta1) * grads[key]
            self.m_hat[key] = self.m[key]/(1 - self.beta1 * self.beta1)

            self.v[key] = self.beta2 * self.v[key] + \
                (1 - self.beta2) * grads[key] * grads[key]
            self.v_hat[key] = self.v[key] / (1 - self.beta2 * self.beta2)

            params[key] -= self.lr * self.m_hat[key] / \
                np.sqrt(self.v_hat[key] + 10e-8)


"""---------------------------------train image--------------------------------------"""
print("loading train image")
img, label = next(training_generator)

train_x = img.reshape(16858, 30000)  # flatten(1차원배열로변경)
train_label = label

"""---------------------------------Test image--------------------------------------"""
print("loading test image")
Timg, Tlabel = next(test_generator)

test_x = Timg.reshape(batch_size, 30000)
test_label = Tlabel

"""--------------------------------NN학습--------------------------------------"""

network_adam = TwoLayerNet(input_size=30000, hidden_size=1000,
                           output_size=33)  # adam
network_sgd = TwoLayerNet(
    input_size=30000, hidden_size=1000, output_size=33)  # sgd
Adam_g = Adam()

# 하이퍼 파라미터
iters_num = 500  # 반복횟수
train_size = train_x.shape[0]
learning_rate = 0.01  # 학습률

train_cost_adam_list = []  # Adam cost list
train_cost_sgd_list = []  # sgd cost list

train_adam_acc_list = []  # train accuracy list
test_adam_acc_list = []  # test accuracy list

iter_per_epoch = max(train_size / batch_size, 1)
"""---------------------------------learning start--------------------------------------"""
start = time.time()  # 시간측정


for i in range(iters_num):

    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = train_x[batch_mask]
    t_batch = train_label[batch_mask]

    # 오차 역전파법
    grad_adam = network_adam.gradient(x_batch, t_batch)
    grad_sgd = network_sgd.gradient(x_batch, t_batch)

    # Adam
    Adam_g.update(network_adam.params, grad_adam)

    # SGD
    for key in network_sgd.params.keys():
        network_sgd.params[key] -= learning_rate * grad_sgd[key]

    # 학습경과기록
    cost = network_adam.cost(x_batch, t_batch)
    cost2 = network_sgd.cost(x_batch, t_batch)
    train_cost_adam_list.append(cost)
    train_cost_sgd_list.append(cost2)
    print("epoch", i, "Adam cost:", cost, "SGD cost2", cost2)

    if(i % 10 == 0):
        train_acc = network_adam.accuracy(x_batch, t_batch, True)
        test_acc = network_adam.accuracy(test_x, test_label, False)
        train_adam_acc_list.append(train_acc)
        test_adam_acc_list.append(test_acc)

        print(f'{i + 1} Train Acc: ', round(train_acc, 3))
        print(f'{i + 1} Test Acc: ', round(test_acc, 3))
        print()


# timer

print("time: ", time.time() - start)

# accuracy plotting

x = np.arange(0, iters_num, 10)
plt.plot(x, train_adam_acc_list, marker='o', markersize=1, label='train acc')
plt.plot(x, test_adam_acc_list, marker='s', markersize=1,
         label='test acc', linestyle='--')
plt.xlabel("iter_num")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# cost plotting

cx = np.arange(0, iters_num, 1)
plt.plot(cx, train_cost_adam_list, marker='o',  label='Adam')
plt.plot(cx, train_cost_sgd_list, marker='s', label='SGD', linestyle='--')
plt.xlabel("iter_num")
plt.ylabel("cost")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


# numerical vs backpropagation 검증
# img, label = next(training_generator)

# train_x = img.reshape(batch_size, 30000)  # flatten(1차원배열로변경)
# train_label = label

# batch_x = train_x[:1]
# batch_t = train_label[:1]

# print(batch_t)

# grad_numerical = network_adam.numerical_gradient(batch_x, batch_t)
# grad_backprop = network_adam.gradient(batch_x, batch_t)

# for key in grad_numerical.keys():
#     diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
#     print(key + ":" + str(diff))
