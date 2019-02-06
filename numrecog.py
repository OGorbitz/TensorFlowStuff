import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

(training_set, training_values), (testing_set, testing_values) = mnist.load_data()


training_set = training_set / 255.0
testing_set = testing_set / 255.0

model = keras.Sequential([
    keras.layers.Conv2D(kernel_size=3, filters=2),
    keras.layers.MaxPool1D(2),
    keras.layers.Conv2D(kernel_size=3, filters=2),
    keras.layers.MaxPool1D(2),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.sigmoid)
])

model.compile(optimizer='adam',
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

plt.figure(figsize=(10,10))
plt.suptitle("Predictions:",y=0.99,fontsize=24)
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imgs[i], cmap=plt.cm.binary)
    plt.xlabel(pred[i])
plt.show()