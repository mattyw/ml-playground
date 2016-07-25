from sklearn import datasets
iris = datasets.load_iris()
import tensorflow as tf
from tensorflow.contrib import learn

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier()
classifier = learn.DNNClassifier(hidden_units=[10,20,10], n_classes=3)

classifier.fit(X_train, y_train, steps=200)

predictions = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
