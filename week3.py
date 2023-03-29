import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

data_size=100

perch_length=np.random.randint(80,440,(1,data_size))/10 #(1,100)

perch_weight=perch_length**2-20*perch_length+110+np.random.randn(1,data_size)*50+100 #(1,100)

plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

train_input,  test_input, train_target, test_target = train_test_split(perch_length.T, perch_weight.T, random_state=42)

print("학습데이터Shape: ", train_input.shape,"테스트데이터Shape: ", test_input.shape)

train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

knr = KNeighborsRegressor()
#knr.n_neighbors = 3
knr.fit(train_input, train_target)
print("--------정확도---------")
#print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))
test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
print("---------오차---------")
print(mae)


