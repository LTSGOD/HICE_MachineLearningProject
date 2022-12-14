import time
import os
from fruitNN import TwoLayerNet
import matplotlib.pyplot as plt
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

fruit_name = ["Apple Braeburn", "Apple Granny", "Apricot", "Avocado", "Banana", "Blueberry", "Cactus fruit", "Cantaloupe", "Cherry", "Clementine", "Corn", "Cucumber Ripe", "Grape blue", "Kiwi", "Lemon", "Limes",
              "Mango", "Onion White", "Orange", "Papaya", "Passion Fruit", "Peach", "Pear", "Pepper green", "Pepper red", "Pineapple", "Plum", "Pomegranate", "Potato Red", "Raspberry", "Strawberry", "Tomato", "Watermelon"]

warnings.simplefilter(action='ignore', category=FutureWarning)

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


# plt.figure(figsize=(100, 100))

# for i in range(100):
#     plt.subplot(10, 10, i+1)
#     plt.imshow(Timg[i])
#     con = name_converter(test_label[i])
#     plt.title(fruit_name[con])
#     plt.axis('off')


# plt.show()

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

test_x = Timg.reshape(batch_size, 30000)  # flatten(1차원배열로변경)
test_label = Tlabel

"""--------------------------------NN학습--------------------------------------"""

network = TwoLayerNet(input_size=30000, hidden_size=1000, output_size=33)
Adam_g = Adam()

# 하이퍼 파라미터
iters_num = 500  # 반복횟수
train_size = train_x.shape[0]
learning_rate = 0.01  # 학습률

train_cost_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
"""---------------------------------learning start--------------------------------------"""
start = time.time()  # 시간측정


for i in range(iters_num):

    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = train_x[batch_mask]
    t_batch = train_label[batch_mask]

    # 오차 역전파법
    grad = network.gradient(x_batch, t_batch)

    # Adam
    Adam_g.update(network.params, grad)

    # SGD
    # for key in network.params.keys():
    #     network.params[key] -= learning_rate * grad[key]

    # 학습경과기록
    cost = network.cost(x_batch, t_batch)
    train_cost_list.append(cost)
    print("epoch", i, "cost:", cost)

    # 1에폭당 정확도
    if(i % 10 == 0):
        train_acc = network.accuracy(x_batch, t_batch, True)
        test_acc = network.accuracy(test_x, test_label, False)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f'{i + 1} Train Acc: ', round(train_acc, 3))
        print(f'{i + 1} Test Acc: ', round(test_acc, 3))
        print()


print("time: ", time.time() - start)
# accricay plotting
x = np.arange(0, iters_num, 10)
plt.plot(x, train_acc_list, marker='o', markersize=2, label='train acc')
plt.plot(x, test_acc_list, marker='s', markersize=2,
         label='test acc', linestyle='--')
plt.xlabel("iter_num")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


""" numerical vs backpropagation 검증
img, label = next(training_generator)

train_x = img.reshape(batch_size, 30000)  # flatten(1차원배열로변경)
train_label = label

batch_x = train_x[:1]
batch_t = train_label[:1]

print(batch_t)

grad_numerical = network.numerical_gradient(batch_x, batch_t)
grad_backprop = network.gradient(batch_x, batch_t)

print("hohoho")

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))"""
