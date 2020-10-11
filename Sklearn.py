import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


df = pd.read_csv("breast-cancer-wisconsin.data")
df.replace('?', -999999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train,y_train)
clf_knn.predict(X_test)

clf_svm = svm.SVC(gamma='auto')
clf_svm.fit(X_train,y_train)
clf_svm.predict(X_test)

accuracy_svm = clf_svm.score(X_test, y_test)
accuracy_knn = clf_knn.score(X_test, y_test)
print('Accuracy of knn:',accuracy_knn)
print('Accuracy of svm:',accuracy_svm)