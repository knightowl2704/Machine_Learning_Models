import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

dataset = pd.read_csv("iris.csv")
print(dataset)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
y=y.reshape(-1,1)
print(y)
#onehotencoder = OneHotEncoder()
#y = onehotencoder.fit_transform(y).toarray()

#print(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train , y_test = train_test_split(X,y,test_size= 1/3, random_state = 0) 

from sklearn.svm import SVC
classifier = SVC(kernel = "linear")
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#confusion = metrics.confusion_matrix(y_test.argmax(axis = 1),y_pred.argmax(axis = 1))
confusion = metrics.confusion_matrix(y_test.flatten(),y_pred.flatten())

print(confusion)

plt.scatter(y_pred, classifier.predict(X_test))
plt.show()
