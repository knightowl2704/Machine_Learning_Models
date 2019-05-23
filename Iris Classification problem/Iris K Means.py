import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

dataset = pd.read_csv("iris.csv")
print(dataset)

X = dataset.iloc[:,[0,1,2,3]].values

#WcSS
wcss=[]
from sklearn.cluster import KMeans
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++", max_iter = 300, n_init = 10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.show()    

kmeans = KMeans(n_clusters =3, init = "k-means++",max_iter = 300,n_init = 10)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

plt.legend()
plt.show()

#print(y)

#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train , y_test = train_test_split(X,y,test_size= 1/3, random_state = 0) 


#from sklearn import metrics

#print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#confusion = metrics.confusion_matriX(y_test.argmaX(aXis = 1),y_pred.argmaX(aXis = 1))
#confusion = metrics.confusion_matriX(y_test.flatten(),y_pred.flatten())

#print(confusion)

#plt.scatter(y_pred, classifier.predict(X_test))
#plt.show()
