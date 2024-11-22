import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv('iris.csv')
x = df.iloc[:, :4].values 


inertia = []
# silhouette_scores = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(x)
    inertia.append(kmeans.inertia_) 
    # silhouette_scores.append(silhouette_score(x, kmeans.labels_)) 

plt.figure(figsize=(10, 5))
plt.plot(K, inertia, 'o-', label='Inertia')
plt.xlabel('Số cụm (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method để tìm K tối ưu')
plt.grid(True)
plt.legend()
plt.show()

# plt.figure(figsize=(10, 5))
# plt.plot(K, silhouette_scores, 'o-', label='Silhouette Score', color='green')
# plt.xlabel('Số cụm (K)')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score để tìm K tối ưu')
# plt.grid(True)
# plt.legend()
# plt.show()

