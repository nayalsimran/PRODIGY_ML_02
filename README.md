### **Importing Libraries**

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.cluster import KMeans
import os
import warnings
warnings.filterwarnings('ignore')

### **Reading the Dataset**

df= pd.read_csv('/Mall_Customers.csv')
df.head()

### **Data Exploration**

df.info()

df.rename(index=str, columns={"Annual Income (k$)": "A_Income", "Spending Score (1-100)": "Score"}, inplace= True)
print(df)

new_data = df.drop(['CustomerID','Gender'], axis=1)


new_data.head()

df.describe()

df.isnull().sum()

plt.figure(figsize=(8, 6))
sns.displot(df["Age"])
plt.xlabel("Age")
plt.show()

### **KMeans with 1 Cluster**

km1 =KMeans(n_clusters = 1).fit(new_data)

new_data['Labels'] =km1.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(x= new_data['A_Income'], y = new_data['Score'], hue=new_data['Labels'],
                palette=sns.color_palette('hls', 3))
plt.title('KMeans with 1 Cluster')
plt.show()

### **KMeans with 3 Clusters**

km3 =KMeans(n_clusters = 3).fit(new_data)

new_data['Labels'] =km3.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(x= new_data['A_Income'], y = new_data['Score'], hue=new_data['Labels'],
                palette=sns.color_palette('hls', 3))
plt.title('KMeans with 3 Clusters')
plt.show()

### **KMeans with 5 Clusters**

km5 =KMeans(n_clusters = 5).fit(new_data)

new_data['Labels'] =km3.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(x= new_data['A_Income'], y = new_data['Score'], hue=new_data['Labels'],
                palette=sns.color_palette('hls', 5))
plt.title('KMeans with 5 Clusters')
plt.show()

### **Bar Chart Plot of Annual Income**

fig, axes= plt.subplots(figsize=(8,6))
sns.barplot(x='Labels', y='A_Income', data=new_data)
axes.set_title("Labels According to Annual Income")
plt.show()

### **Bar Chart Plot of Scoring History**

fig, axes= plt.subplots(figsize=(8,6))

sns.barplot(x='Labels', y='Score', data=new_data)
axes.set_title("Labels According to Scoring History")

plt.show()

### **Clustering using KMeans**

### **Segementation Using Age and Spending Score**

X1 = df[['Age' , 'Score']].iloc[: , :].values
inertia = []
for n in range(1, 11):
  algorithm= (KMeans(n_clusters = n ,init='k-means++', n_init=10 ,max_iter=100,
                     tol=0.0001, random_state= 111 , algorithm='elkan'))
  algorithm.fit(X1)
  inertia.append(algorithm.inertia_)

plt.figure(1, figsize = (12,6))
plt.plot(np.arange(1, 11), inertia , 'o')
plt.plot(np.arange(1, 11), inertia , '-', alpha = 0.5)
plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
plt.show()



### **Segmenation Using Annual Income and Spending Score**

X2 = df[['A_Income' , 'Score']].iloc[: , :].values
inertia = []
for n in range(1, 11):
  algorithm= (KMeans(n_clusters = n ,init='k-means++', n_init=10 ,max_iter=100,
                     tol=0.0001, random_state= 111 , algorithm='elkan'))
  algorithm.fit(X2)
  inertia.append(algorithm.inertia_)

plt.figure(1, figsize = (12,6))
plt.plot(np.arange(1, 11), inertia , 'o')
plt.plot(np.arange(1, 11), inertia , '-', alpha = 0.5)
plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
plt.show()
