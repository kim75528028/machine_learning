import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


ep = np.array(pd.read_excel('./ep.xlsx'))
fp = np.array(pd.read_excel('./fp.xlsx'))

plt.plot(fp[:,0])
plt.plot(fp[:,1])
plt.plot(fp[:,2])
plt.plot(ep*10000)
#plt.show()


fp = fp[:,0:3]

poly = PolynomialFeatures(degree=3, include_bias=False)
poly.fit(fp)
train_poly = poly.transform(fp)
print(np.shape(train_poly))