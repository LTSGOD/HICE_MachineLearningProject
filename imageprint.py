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
training_dir = "C:\\Users\\82104\\OneDrive\\HICEMachingLearningProject\\archive\\train\\train\\train1"
# test data 파일경로
test_dir = "C:\\Users\\82104\\OneDrive\\HICEMachingLearningProject\\archive\\test"

training_datagen = ImageDataGenerator()  # 정규화

test_datagen = ImageDataGenerator(rescale=1./255)  # 정규화

batch_size = 100  # mini_batch 이용

training_generator = training_datagen.flow_from_directory(
    training_dir,
    batch_size=738,  # batch size
    target_size=(100, 100),  # target 크기 100 x 100
    class_mode='categorical',  # one hot encoding 사용
)

print("loading train image")
img, label = next(training_generator)
print(img[0].shape)
print(img[0])
for i in range(100):
    print()
    for j in range(100):
        print(end='|')
        for k in range(3):
            print(img[0][i][j][k], end=' ')
train_x = img.reshape(738, 30000)  # flatten(1차원배열로변경)
train_label = label
