import sklearn
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import datasets
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
#Mempersiapkan Data
data1 = pd.read_csv ("DataAir.csv")
#Data Scaling
x = data1[data1.columns[:3]]
y = data1['Kelas']
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
#Split Data
data1 = load_iris()
x = data1.data
y = data1.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2)
#KNN K = 3
KNN = neighbors.KNeighborsClassifier(n_neighbors = 3, weights ='uniform')
KNN.fit(x_train, y_train)
predictionKNN = KNN.predict(x_test)
accuracyKNN = metrics.accuracy_score(y_test, predictionKNN)
print('=============== METODE KNN dengan K=3 =================')
print('prediction: ', predictionKNN)
print('actual: ', y_test)
print('accuracy: ', accuracyKNN)
#KNN K = 5
KNN = neighbors.KNeighborsClassifier(n_neighbors = 5, weights ='uniform')
KNN.fit(x_train, y_train)
predictionKNN = KNN.predict(x_test)
accuracyKNN = metrics.accuracy_score(y_test, predictionKNN)
print('=============== METODE KNN dengan K=5 =================')
print('prediction: ', predictionKNN)
print('actual: ', y_test)
print('accuracy: ', accuracyKNN)
#KNN K = 7
KNN = neighbors.KNeighborsClassifier(n_neighbors = 7, weights ='uniform')
KNN.fit(x_train, y_train)
predictionKNN = KNN.predict(x_test)
accuracyKNN = metrics.accuracy_score(y_test, predictionKNN)
print('=============== METODE KNN dengan K=7 =================')
print('prediction: ', predictionKNN)
print('actual: ', y_test)
print('accuracy: ', accuracyKNN)
#KNN K = 9
KNN = neighbors.KNeighborsClassifier(n_neighbors = 9, weights =
'uniform')
KNN.fit(x_train, y_train)
predictionKNN = KNN.predict(x_test)
accuracyKNN = metrics.accuracy_score(y_test, predictionKNN)
print('=============== METODE KNN dengan K=9 =================')
print('prediction: ', predictionKNN)
print('actual: ', y_test)
print('accuracy: ', accuracyKNN)