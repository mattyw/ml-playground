#import tensorflow as tf
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import random
from tensorflow.contrib import learn
import csv


def read_data(filename):
    data = []
    for row in csv.DictReader(open(filename)):
        data.append(row)
    return data

def get_features(data, keys):
    features = []
    labels = []
    for row in data:
        passenger = row_to_passenger(row, keys)
        features.append(passenger)
        labels.append(int(row["Survived"]))
    return (features, labels)

AVG_AGE=0

def row_to_passenger(row, keys):
    passenger = []
    for key in keys:
        f = row[key]
        if key == "Fare":
            if f == "":
                f = 10
            f = float(f)
        if key == "Pclass":
            f = int(f)*33
        if key == "Age":
            if f == "":
                f = 40
            else:
                f = float(row["Age"])
        if key == "Sex":
            if f == "male":
                f = Male
            if f == "female":
                f = Female
        passenger.append(float(f))
    return passenger


def features_from_test(data, keys):
    features = []
    for row in data:
        passenger = row_to_passenger(row, keys)
        features.append(passenger)
    return features

def kaggle_output_format(data, predictions):
    output = []
    output.append(["PassengerId","Survived"])
    if len(data) != len(predictions):
        raise
    i = 0
    for row in data:
        #print(row, predictions[i])
        output.append([row, predictions[i]])
        i += 1
    with open('kaggle_output.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output)
    survive = len([x[1]for x in output if x[1] == 1])
    died = len([x[1] for x in output if x[1] == 0])
    print(survive, died, survive+died)
    return output

# Data Clean
Male = 0
Female = 100

class CustomClassifier():
    def __init__(self):
        pass

    def fit(self, data, labels, **kwargs):
        pass

    def predict(self, features):
        prediction = []
        for row in features:
            survived = 0
            if row[1] == Female:
                survived = 1
                if row[0] == 3:
                    survived = 0
            else:
                survived = 0 
            prediction.append(survived)
        return prediction

if __name__ == '__main__':
    feature_keys = ["Sex", "Pclass", "Fare", "Parch"]
    train = read_data("train.csv")
    (features, labels) = get_features(train, feature_keys)
    #for f, l in zip(features, labels):
        #print(f, l)

    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = .4, random_state=0)

    #from sklearn.neighbors import KNeighborsClassifier
    #classifier = KNeighborsClassifier()
    #classifier = tree.DecisionTreeClassifier(max_depth=10)
    #classifier = svm.SVC(kernel="linear")
    #classifier = learn.DNNClassifier(hidden_units=[20,40,20], n_classes=2, model_dir="titanic_model")
    #classifier = CustomClassifier()
    #classifier = RandomForestClassifier(n_estimators=100)
    #classifier = AdaBoostClassifier(n_estimators=100)
    classifier = GradientBoostingClassifier(n_estimators=50)

    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=int)

    #classifier.fit(X_train, y_train, steps=10000)
    classifier.fit(X_train, y_train)

    X_test = np.array(X_test, dtype=float)

    predictions = classifier.predict(X_test)

    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, predictions))

    # Do Kaggle test
    print("kaggle")
    test = read_data("test.csv")
    passengerIds = [k["PassengerId"] for k in test]
    kaggle_test = features_from_test(test, feature_keys)

    kaggle_test = np.array(kaggle_test, dtype=float)
    predictions = classifier.predict(kaggle_test)
    kaggle_output_format(passengerIds, predictions)


