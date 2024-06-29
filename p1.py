#KMean Clustering Project
#Data Preprocessing
#EDA
#finding the optimal number of clusters using elbow method

#import the lib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#load the data
data=pd.read_csv("Mall_Customers.csv")

print(data.head())

#Preprocessing
print(data.shape)

print(data.isnull().sum())
print(data.duplicated().sum())

print(data.info())
print(data.describe())

data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})


#EDA
sns.set()
sns.pairplot(data,hue='Gender')
plt.show()

corr =data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

plt.figure(figsize=(6,6))
sns.histplot(data['Age'])
plt.show()


plt.figure(figsize=(6,6))
sns.histplot(data['Annual Income (k$)'])
plt.show()

plt.figure(figsize=(6,6))
sns.histplot(data['Spending Score (1-100)'])
plt.show()

plt.figure(figsize=(6,6))
sns.histplot(data['Gender'])
plt.show()

x=data.iloc[:,[3,4]].values
print(x)

# Kmeans Clustering
wcss=[]
for i in range(1,11):
    k_means=KMeans(n_clusters=i,init='k-means++',random_state=42)
    k_means.fit(x)

    wcss.append(k_means.inertia_)
print(wcss)


plt.plot(range(1,11),wcss)
plt.show()


#Training the K-means Clustering Model
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
y=kmeans.fit_predict(x)
print(y)



#plotting all the clusters and their centroids

plt.figure(figsize=(8, 8))
plt.scatter(x[y == 0, 0], x[y == 0, 1], s=50, c='green', label='cluster1')
plt.scatter(x[y == 1, 0], x[y == 1, 1], s=50, c='red', label='cluster2')
plt.scatter(x[y == 2, 0], x[y == 2, 1], s=50, c='yellow', label='cluster3')
plt.scatter(x[y == 3, 0], x[y == 3, 1], s=50, c='violet', label='cluster4')
plt.scatter(x[y == 4, 0], x[y == 4, 1], s=50, c='blue', label='cluster5')

# Corrected the attribute name
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='centroids')

plt.title('Customer Group')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()  # Added legend to display the labels
plt.show()
