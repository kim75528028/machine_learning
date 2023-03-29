import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()
#print(pd.unique(fish['Species']))

fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
fish_target = fish['Species'].to_numpy()
#print(fish_input[:5])

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

ss = StandardScaler() # 표준점수 만들기
ss.fit(train_input)
train_scaled = ss.transform(train_input) # 표준화된 점수로 테스티 및 학습용 데이터 제작
test_scaled = ss.transform(test_input)

kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
"""
print(kn.classes_)
print('---------------------------------')
print(kn.predict(test_scaled[:5]))
print('---------------------------------')
"""
proba = kn.predict_proba(test_scaled[:5])
"""
print(kn.score(train_scaled,train_target))
print('---------------------------------')
print(test_scaled,test_target)
print('---------------------------------')
print(np.round_(proba, decimals=4)) # proba = 직군 결정이 아닌 확률값으로 표시


bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]
"""
from sklearn.linear_model import LogisticRegression
""" 이진 분류
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt) # 모델 제작

print(lr.predict(train_bream_smelt[:5]))
print('\n---------------------------------\n') #이진 분류
print(lr.predict_proba(train_bream_smelt[:5])) #확률 (0과 1 사이의 값을 표시)

print(lr.coef_, lr.intercept_) #계수 확인

decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions) #z값

from scipy.special import expit
print(expit(decisions)) #sigmod 통과 결과


#다중 분류

lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)

#print(lr.score(train_scaled, train_target))
#print(lr.score(test_scaled, test_target))
#print('\n---------------------------------\n')

proba = lr.predict_proba(test_scaled[:5])
#print(np.round(proba, decimals=3))
#print('\n---------------------------------\n')
#print(lr.coef_.shape, lr.intercept_.shape)

# 소프트 맥스
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))

from scipy.special import softmax

proba = softmax(decision, axis=1)
print('\n---------------------------------\n')
print(np.round(proba, decimals=3))
"""
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss='log', max_iter=20, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled,train_target))
print(sc.score(test_scaled, test_target))
print('\n---------------------------------\n')

sc.partial_fit(train_scaled, train_target)

print(sc.score(train_scaled,train_target))
print(sc.score(test_scaled, test_target))

