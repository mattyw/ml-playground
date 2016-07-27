#import tensorflow as tf
from sklearn import tree
from sklearn import svm
#import numpy as np
import csv


def read_data(filename):
    train_data = csv.reader(open(filename))
    header = next(train_data)  # Skip the header in the csv file
    i = 0
    features = {}
    for title in header:
        features[title] = i
        i = i + 1
    data = []
    for row in train_data:
        data.append(row)

    return (features, data)

def get_features(data, survived_idx, feature_indexes):
    features = []
    labels = []
    for row in data:
        passenger = row_to_passenger(row, feature_indexes)
        features.append(passenger)
        labels.append(int(row[survived_idx]))
    return (features, labels)


def row_to_passenger(row, feature_indexes):
    passenger = []
    for idx in feature_indexes:
        f = row[idx]
        if f == "male":
            f = Male
        if f == "female":
            f = Female
        if f == "":
            f = 0
        passenger.append(float(f))
    return passenger


def features_from_test(data, feature_indexes):
    features = []
    for row in data:
        passenger = row_to_passenger(row, feature_indexes)
        features.append(passenger)
    return features

def kaggle_output_format(data, predictions):
    output = []
    output.append(["PassengerId","Survived"])
    if len(data) != len(predictions):
        return []
    i = 0
    for row in data:
        output.append([row[0], predictions[i]])
        i += 1
    with open('kaggle_output.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output)
    return output

# Targets
SURVIVED = 1
DIED = 0

# Data Clean
Male = 0
Female = 1

if __name__ == '__main__':
    (fm, train) = read_data("train.csv")
    (features, labels) = get_features(train, fm["Survived"],
        [fm["Pclass"], fm["Sex"], fm["Age"], fm["SibSp"], fm["Parch"]
    ])


    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = .5)

    #from sklearn.neighbors import KNeighborsClassifier
    #classifier = KNeighborsClassifier()
    #classifier = tree.DecisionTreeClassifier()
    classifier = svm.SVC(kernel="linear")

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, predictions))

    # Do Kaggle test
    (test_fm, test) = read_data("test.csv")
    kaggle_test = features_from_test(test,
        [test_fm["Pclass"], test_fm["Sex"], test_fm["Age"], test_fm["SibSp"], test_fm["Parch"]])

    #classifier = svm.SVC(kernel="linear")
    classifier = tree.DecisionTreeClassifier()

    classifier.fit(features, labels)

    predictions = classifier.predict(kaggle_test)
    kaggle_output_format(test, predictions)


