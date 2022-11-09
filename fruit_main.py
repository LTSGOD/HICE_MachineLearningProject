import numpy as np
import warnings
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
from fruitNN import TwoLayerNet
import os

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
    batch_size=batch_size,  # batch size
    target_size=(100, 100),  # target 크기 100 x 100
    class_mode='categorical',  # one hot encoding 사용
    shuffle=True  # 무작위
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    batch_size=batch_size,
    class_mode='categorical',
    target_size=(100, 100),  # target 크기 100 x 100
)


"""def name_converter(par):
    a = 0
    for i in range(33):
        if(par[i] == 1):
            return a
        else:
            a += 1


plt.figure(figsize=(100, 100))

for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.imshow(Timg[i])
    con = name_converter(test_label[i])
    plt.title(fruit_name[con])
    plt.axis('off')

plt.show()"""


"""--------------------------------NN학습--------------------------------------"""

network = TwoLayerNet(input_size=30000, hidden_size=2000, output_size=33)

# 하이퍼 파라미터
iters_num = 10000
train_size = 16874  # train_x.shape[0]
learning_rate = 0.1

train_cost_list = []
train_acc_list = []
test_acc_list = []

# 1epoch당 반복수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    img, label = next(training_generator)
    #Timg, Tlabel = next(test_generator)

    train_x = img.reshape(batch_size, 30000)  # flatten(1차원배열로변경)
    train_label = label

    #test_x = Timg.reshape(batch_size, 30000)
    #test_label = Tlabel

    # 오차 역전파법
    grad = network.gradient(train_x, train_label)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습경과기록
    cost = network.cost(train_x, train_label)
    train_cost_list.append(cost)

    # 1에폭당 정확도
