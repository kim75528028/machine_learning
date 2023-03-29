import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


ep = pd.read_excel('./ep.xlsx')
fp = pd.read_excel('./fp.xlsx')
target_data = ep.to_numpy()
train_data = fp.to_numpy()

train_input, test_input, train_target, test_target = train_test_split(
    train_data, target_data, random_state=1)

poly = PolynomialFeatures(include_bias=False)

poly.fit(train_input)
train_poly = poly.transform(train_input)

#print(train_poly.shape)

poly.get_feature_names_out()
['x0','x1','x2','x0^2','x0^1','x0 x2','x0 x2', 'x1^2', 'x1 x2', 'x2^2']
test_poly = poly.transform(test_input)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.score(train_poly, train_target))

print(lr.score(test_poly, test_target))

poly = PolynomialFeatures(degree = 5, include_bias=False)

poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

print("---------------------------------")

lr.fit(train_poly, train_target)

print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

train_score = []
test_score = []

alpha_list = [0.0001, 0.001, 0.1, 1, 10, 100, 1000]
for alpha in alpha_list:
    rd = Ridge()
    rd.fit(train_scaled, train_target)

    train_score.append(rd.score(train_scaled, train_target))
    test_score.append(rd.score(test_scaled, test_target))

rd = Ridge(alpha = 1)
rd.fit(train_scaled, train_target)
print("릿지")
print(rd.score(train_scaled,train_target))
print(rd.score(test_scaled, test_target))

from sklearn.linear_model import Lasso

ls = Lasso()
ls.fit(train_scaled, train_target)

train_score = []
test_score = []

alpha_list = [0.0001, 0.001, 0.1, 1, 10, 100, 1000]
for alpha in alpha_list:
    ls = Lasso()
    ls.fit(train_scaled, train_target)

    train_score.append(rd.score(train_scaled, train_target))
    test_score.append(rd.score(test_scaled, test_target))

ls = Lasso(alpha = 1)
ls.fit(train_scaled, train_target)
print('라쏘')
print(ls.score(train_scaled, train_target))
print(ls.score(test_scaled, test_target))