import numpy as np
import matplotlib.pyplot as plt

fruits = np.load('fruits_300.npy')
# print(fruits.shape)

apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)

# print(apple.shape)

#plt.hist(np.mean(apple, axis=1), alpha=0.8)
#plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
#plt.hist(np.mean(banana, axis=1), alpha=0.8)
#plt.legend(['apple', 'pineapple', 'banana'])
#plt.show()

#fig, axs = plt.subplots(1, 3, figsize=(20, 5))
#axs[0].bar(range(10000), np.mean(apple, axis=0))
#axs[1].bar(range(10000), np.mean(pineapple, axis=0))
#axs[2].bar(range(10000), np.mean(banana, axis=0))
#plt.show()

apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)

#fig, axs = plt.subplots(1, 3, figsize=(20, 5))
#axs[0].imshow(apple_mean, cmap='gray_r')
#axs[1].imshow(pineapple_mean, cmap='gray_r')
#axs[2].imshow(banana_mean, cmap='gray_r')
#plt.show()

abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1,2))
print(abs_mean.shape)

apple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        axs[i,j].imshow(fruits[apple_index[i*10+j]], cmap='gray_r')

        axs[i,j].axis('off')
#plt.show()

#클러스터
from sklearn.cluster import KMeans
fruits_2d = fruits.reshape(-1, 100*100)

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)

print(km.labels_)

print(np.unique(km.labels_, return_counts=True))
def draw_fruits(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n/10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols,
                            figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')


fruits_2d = fruits.reshape(-1, 100*100)
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca.fit(fruits_2d)

print(pca.components_.shape)

draw_fruits(pca.components_.reshape(-1,100,100))

print(fruits_2d.shape)

fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)