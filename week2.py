import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split

fish_data=np.load('data_input.npy')
fish_target=np.load('data_target.npy')

plt.scatter(fish_data[:500,0],fish_data[:500,1])
plt.scatter(fish_data[500:900,0],fish_data[500:900,1])
plt.scatter(fish_data[900:1300,0],fish_data[900:1300,1])
plt.scatter(fish_data[1300:,0],fish_data[1300:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


kn=KNeighborsClassifier()
kn.fit(fish_data,fish_target) # fit은 학습시키기. (mapping 관계에서 사용) fit(입력데이터,출력데이터)
#print(kn.score(fish_data,fish_target))
#print(kn.predict([[25,25]]))

input_arr=np.array(fish_data)
target_arr=np.array(fish_target)

index= np.arange(1800)
np.random.shuffle(index)

train_input = input_arr[index[:1350]]
train_target = target_arr[index[:1350]]

test_input = input_arr[index[1350:]]
test_target = target_arr[index[1350:]]

#plt.scatter(train_input[:, 0], train_input[:, 1])
#plt.xlabel("X")
#plt.ylabel("Y")
#plt.show()

mean=np.mean(train_input, axis=0)
std=np.std(train_input,axis=0)

print(mean,std)

train_scaled= (train_input-mean)/std
new = ([21.5, 450]-mean)/std

kn.fit(train_scaled, train_target)

plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0],new[1],marker='^')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


distances, indexes = kn.kneighbors([new])

plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new[0],new[1],marker='^')
plt.scatter(train_scaled[indexes,0],
            train_scaled[indexes,1],marker='D')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


kn.n_neighbors = 10
distances, indexes = kn.kneighbors([new])

plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new[0],new[1],marker='^')
plt.scatter(train_scaled[indexes,0],
            train_scaled[indexes,1],marker='D')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


kn.n_neighbors = 30
distances, indexes = kn.kneighbors([new])

plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new[0],new[1],marker='^')
plt.scatter(train_scaled[indexes,0],
            train_scaled[indexes,1],marker='D')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


kn.n_neighbors = 50
distances, indexes = kn.kneighbors([new])

plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new[0],new[1],marker='^')
plt.scatter(train_scaled[indexes,0],
            train_scaled[indexes,1],marker='D')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

