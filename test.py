import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
import warnings
import os

fruit_name = ["Apple Braeburn", "Apple Granny", "Apricot", "Avocado", "Banana", "Blueberry", "Cactus fruit", "Cantaloupe", "Cherry", "Clementine", "Corn", "Cucumber Ripe", "Grape blue", "Kiwi", "Lemon", "Limes",
              "Mango", "Onion White", "Orange", "Papaya", "Passion Fruit", "Peach", "Pear", "Pepper green", "Pepper red", "Pineapple", "Plum", "Pomegranate", "Potato Red", "Raspberry", "Strawberry", "Tomato", "Watermelon"]

warnings.simplefilter(action='ignore', category=FutureWarning)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# trainig data 파일경로
training_dir = "C:\\Users\\82104\\OneDrive\\HICEMachingLearningProject\\archive\\train\\train"
# test data 파일경로
test_dir = "C:\\Users\\82104\\OneDrive\\HICEMachingLearningProject\\archive\\test"

training_datagen = ImageDataGenerator(rescale=1./255)  # 정규화

test_datagen = ImageDataGenerator(rescale=1./255)  # 정규화

batch_size = 100  # mini_batch 이용

training_generator = training_datagen.flow_from_directory(
    training_dir,
    batch_size=16858,  # 2393,  # batch size 16858
    target_size=(100, 100),  # target 크기 100 x 100
    class_mode='sparse',  # one hot encoding 사용
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    batch_size=batch_size,
    class_mode='sparse',
    target_size=(100, 100),  # target 크기 100 x 100
)

"""---------------------------------train image--------------------------------------"""
print("loading train image")
img, label = next(training_generator)

# train_x = img.reshape(2393, 30000)  # flatten(1차원배열로변경)
# train_x = img
train_label = label

"""---------------------------------Test image--------------------------------------"""
print("loading test image")
Timg, Tlabel = next(test_generator)

test_x = Timg.reshape(batch_size, 30000)
test_label = Tlabel

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(100, 100, 3)))
model.add(keras.layers.Dense(1000, activation='relu', name='hidden1'))
model.add(keras.layers.Dense(33, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(img, train_label, epochs=5)
model.evaluate(Timg, test_label)
