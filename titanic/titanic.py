import csv
import numpy as np

train_data = csv.reader(open("train.csv"))
header = next(train_data) #Skip the header in the csv file

data = []
for row in train_data:
    data.append(row)
data = np.array(data)
print(data[0::,3])
number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers
print(proportion_survivors)

women_only_stats = data[0::,4] == "female"
men_only_stats = data[0::,4] != "female"

women_onboard = data[women_only_stats,1].astype(np.float)     
men_onboard = data[men_only_stats,1].astype(np.float)
proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)  
proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard) 

print('Proportion of women who survived is %s' % proportion_women_survived)
print('Proportion of men who survived is %s' % proportion_men_survived)

"""
    Create a very simple prediction that all women survived.
"""
def gender_based_prediction():
    test_file = open('test.csv')
    test_file_object = csv.reader(test_file)
    header = next(test_file_object)
    prediction_file = open("genderbasedmodel.csv", "w")
    prediction_file_object = csv.writer(prediction_file)
    prediction_file_object.writerow(["PassengerId", "Survived"])
    for row in test_file_object:       # For each row in test.csv
        if row[3] == 'female':         # is it a female, if yes then                                       
            prediction_file_object.writerow([row[0],'1'])    # predict 1
        else:                              # or else if male,       
            prediction_file_object.writerow([row[0],'0'])    # predict 0
    test_file.close()
    prediction_file.close()
