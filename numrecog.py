import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

(training_set_i, training_values), (testing_set_i, testing_values) = mnist.load_data()

img_rows = 28
img_cols = 28

if keras.backend.image_data_format == 'channels_first':
    training_set = training_set_i.reshape(training_set_i.shape[0], 1, img_rows, img_cols)
    testing_set = testing_set_i.reshape(testing_set_i.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    training_set = training_set_i.reshape(training_set_i.shape[0], img_rows, img_cols, 1)
    testing_set = testing_set_i.reshape(testing_set_i.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

training_set = training_set / 255.0
testing_set = testing_set / 255.0

model = keras.Sequential([
    keras.layers.Conv2D(kernel_size=(3,3), filters=32, activation=tf.nn.relu),
    keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding="same", activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adadelta',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_set, training_values, epochs=10)

num = (np.random.rand(25)*10000).astype(int)

imgs = testing_set[num]
prediction = model.predict(imgs)
pred = []
for i in range(25):
    big=0.0
    index=0
    for j in range(len(prediction[i])):
        if(prediction[i][j]>big):
            big = prediction[i][j]
            index = j;
    pred.append(index)

print(pred[0])
plt.figure(figsize=(10,10))
plt.suptitle("Predictions:",y=0.99,fontsize=24)
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(testing_set_i[num[i]], cmap=plt.cm.binary)
    plt.xlabel(pred[i])
plt.show()