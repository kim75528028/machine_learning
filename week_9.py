import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate
from tensorflow import keras
import tensorflow as tf

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)

# from sklearn.linear_model import SGDClassifier
#
# sc = SGDClassifier(loss='log_loss', max_iter=5, random_state=42)
#
# scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
# print(np.mean(scores['test_score']))

from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,))
dense2 = keras.layers.Dense(10, activation='softmax')
model = keras.Sequential([dense1,dense2])

# model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
# print(train_target[:10])
# model.fit(train_scaled, train_target, epochs=5)
# model.evaluate(val_scaled, val_target)

model = keras.Sequential([
    keras.layers.Dense(100, activation='sigmoid', input_shape=(784,), name='hidden'),
    keras.layers.Dense(10, activation='softmax', name='output')
], name='패션 MNIST 모델')

model.summary()

model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784,)))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

# model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

# model.fit(train_scaled, train_target, epochs=5)


model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

model.fit(train_scaled, train_target, epochs=5)
