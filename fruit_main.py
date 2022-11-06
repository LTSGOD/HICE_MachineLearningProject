import numpy as np
import warnings
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
import os

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

img, label = next(training_generator)
Timg, Tlabel = next(test_generator)

train_x = img.reshape(batch_size, 30000)  # flatten(1차원배열로변경)
train_label = label

test_x = Timg.reshape(batch_size, 30000)
test_label = Tlabel


'''plt.figure(figsize=(20, 20))

for i in range(8):
    plt.subplot(3, 3, i+1)
    plt.imshow(Timg[i])
    plt.title(test_label[i])
    plt.axis('off')

plt.show()'''
#
