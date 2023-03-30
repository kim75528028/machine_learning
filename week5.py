import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

#DataFrame ==> np ==> 표준화(평균을 중심으로 동일한 표준편차)

wine = pd.read_csv('wine.csv')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)

sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

ss = StandardScaler() #표준화 도구
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

lr = LogisticRegression()
lr.fit(train_scaled, train_target)

#print(lr.score(train_scaled, train_target))
#print(lr.score(test_scaled, test_target))
#print(lr.coef_, lr.intercept_)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(train_scaled, train_target)

#print(dt.score(train_scaled, train_target))
#print(dt.score(test_scaled, test_target))

from sklearn.tree import plot_tree

#plt.figure(figsize=(10,7))
#plot_tree(dt)
#plt.show()

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
#print(dt.score(train_input, train_target))
#print(dt.score(test_input,test_target))
#plt.figure(figsize=(20,15))
#plot_tree(dt, filled=True, feature_names=['alcohol', 'suger', 'pH'])
#plt.show()

#print(dt.feature_importances_)

from sklearn.model_selection import cross_validate

score = cross_validate(dt, train_input, train_target)

#print(np.mean(score['test_score']))


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

params = {'min_impurity_decrease':uniform(0.0001,0.001)
          'max_depth': randint(20,50)
          'min_samples_split': randint(2,25)
          'min_samples_leaf': randint(1,25)}
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=1, random_state=42)
gs.fit(train_input, train_target)

print(gs.best_params_)
print("--------------------------")

print(np.max(gs.cv_results_['mean_test_score']))

print("--------------------------")

dt=gs.best_estimator_
print(dt.score(test_input, test_target))