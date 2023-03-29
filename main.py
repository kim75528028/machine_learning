import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

input = np.load('data_input.npy')
target = np.load('data_target.npy')

#print(np.shape(input))
#print(np.shape(target))

# target 4개 , 데이터 정규화 , 데이터 테스트 트레이닝으로 쪼개기

kn = KNeighborsClassifier()
kn.fit(input, target)
#print(kn.score(input,target))

np.random.seed(42)
index = np.arange(1800)
np.random.shuffle(index)


train_input = input_train[index[:1200]]
train_target = target_train[index[:1200]]
test_input = input_test[index[1200:]]
test_target = target_test[index[1200:]]



plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#plt.scatter(input[:, 0], input[:, 1])
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.show()