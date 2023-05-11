import numpy as np
from tensorflow import keras
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


data_size=1000

train_plus_data = np.random.randint(100,size=(data_size,2))+np.random.randint(100,size=(data_size,2))/100
train_minus_data = np.random.randint(100,size=(data_size, 2)) + np.random.randint(100, size=(data_size,2))/100

train_minus_data = -1 * train_minus_data


train_data = np.concatenate((train_plus_data,train_minus_data), axis=0)
model = keras.Sequential()

#덧셈

train_ans = train_data[:,0]+train_data[:,1]


model.add(keras.layers.Dense(20,activation='relu'))
model.add(keras.layers.Dense(10,activation='relu'))
model.add(keras.layers.Dense(5,activation='elu'))
model.add(keras.layers.Dense(1,activation=keras.layers.LeakyReLU(alpha=0.1)))
model.compile(optimizer=tf.keras.optimizers.Adam(0.00009),loss='mse',metrics=['accuracy'])

model.fit(train_data,train_ans,epochs=20,batch_size=1)

z=np.array([10.1,20.3]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)

z=np.array([10,20]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)

z=np.array([10,-20]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)

z=np.array([5,-10]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)

z=np.array([100,200]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)

#뺄셈
train_plus_data[500:, 1] = -1*train_plus_data[500:, 1]
train_minus_data[:500, 1] = -1*train_minus_data[:500, 1]

train_data = np.concatenate((train_plus_data,train_minus_data), axis=0)
train_ans = train_data[:,0] - train_data[:,1]

model.add(keras.layers.Dense(20,activation='relu'))
model.add(keras.layers.Dense(10,activation='relu'))
model.add(keras.layers.Dense(5,activation='elu'))
model.add(keras.layers.Dense(1,activation=keras.layers.LeakyReLU(alpha=0.1)))
model.compile(optimizer=tf.keras.optimizers.Adam(0.00009),loss='mse',metrics=['accuracy'])

model.fit(train_data,train_ans,epochs=20,batch_size=1)

z=np.array([10.1,20.3]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)

z=np.array([10,20]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)

z=np.array([10,-20]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)

z=np.array([5,-10]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)

z=np.array([100,200]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)

#곱셈
train_plus_data[500:, 1] = -1*train_plus_data[500:, 1]
train_minus_data[:500, 1] = -1*train_minus_data[:500, 1]

train_data = np.concatenate((train_plus_data,train_minus_data), axis=0)
train_ans = train_data[:,0] * train_data[:,1]

model.add(keras.layers.Dense(50,activation='relu'))
model.add(keras.layers.Dense(20,activation='relu'))
model.add(keras.layers.Dense(10,activation='relu'))
model.add(keras.layers.Dense(5,activation='linear'))
model.add(keras.layers.Dense(1,activation=keras.layers.LeakyReLU(alpha=0.1)))
model.compile(optimizer=tf.keras.optimizers.Adam(0.00009),loss='mse',metrics=['accuracy'])

model.fit(train_data,train_ans,epochs=20,batch_size=1)


z=np.array([10.1,20.3]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)

z=np.array([10,20]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)

z=np.array([10,-20]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)

z=np.array([5,-10]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)

z=np.array([100,200]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)

#나눗셈
train_plus_data[500:, 1] = -1*train_plus_data[500:, 1]
train_minus_data[:500, 1] = -1*train_minus_data[:500, 1]

train_data = np.concatenate((train_plus_data,train_minus_data), axis=0)
train_ans = train_data[:,0] / train_data[:,1]

model.add(keras.layers.Dense(50,activation='relu'))
model.add(keras.layers.Dense(20,activation='relu'))
model.add(keras.layers.Dense(10,activation='relu'))
model.add(keras.layers.Dense(5,activation='linear'))
model.add(keras.layers.Dense(1,activation=keras.layers.LeakyReLU(alpha=0.1)))
model.compile(optimizer=tf.keras.optimizers.Adam(0.00009),loss='mse',metrics=['accuracy'])

model.fit(train_data,train_ans,epochs=20,batch_size=1)


z=np.array([10.1,20.3]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)

z=np.array([10,20]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)

z=np.array([10,-20]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)

z=np.array([5,-10]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)

z=np.array([100,200]).reshape(1,2)
q=model.predict(z,batch_size=1)
print(q)
#아무리 머리를 써봐도 곱셈과 나눗셈을 할 때 음수계산에 대한 정확도를 올릴 방법을 모르겠어요...