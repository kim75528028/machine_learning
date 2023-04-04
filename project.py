import pandas as pd
import matplotlib.pyplot as plt

ks = pd.read_csv('kospi.csv')
ns = pd.read_csv('nasdaq.csv')
ts = pd.read_excel('trans.xlsx')


plt.plot(ks*5)
plt.plot(ns)
plt.plot(ts*10)
plt.show()