from sklearn import tree

bumpy = 0
smooth = 1

apple = 0
orange = 1

features = [
    [140, smooth],
    [130, smooth],
    [150, bumpy],
    [170, bumpy],
    ]

labels = [
    apple,
    apple,
    orange,
    orange,
    ]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
prediction = clf.predict([[160, bumpy]])
if prediction == 0:
    print("apple")
if prediction == 1:
    print("orange")
