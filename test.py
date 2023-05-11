import numpy as np
import pandas as pd

fish = pd.read_csv('midterm.csv')

fish_input = fish[['Weight','Length','Diagonal','Height']].to_numpy()
fish_target = fish['xSpecies'].to_numpy()
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')

mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
train_scaled = (train_input - mean) / std

mean = np.mean(test_input, axis=0)
std = np.std(test_input, axis=0)
test_scaled = (test_input - mean) / std

train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
print(lr.coef_, lr.intercept_)

decisions = lr.decision_function(train_bream_smelt[:4])
print(decisions)

from scipy.special import expit

print(expit(decisions))

lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

proba = lr.predict_proba(test_scaled[:4])
print(np.round(proba, decimals=3))

print(lr.coef_.shape, lr.intercept_.shape)

decision = lr.decision_function(test_scaled[:4])
print(np.round(decision, decimals=2))

from scipy.special import softmax

proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))
