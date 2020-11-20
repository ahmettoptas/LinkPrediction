import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

desired_width = 320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns', 10)

#mixed datas from networkx.py which we created. We create trainingData csv by using these block.
"""trainingData = pd.read_csv("mixedDataFirstHalf.csv")
testData = pd.read_csv("mixedDataSecHalf.csv")
trainingData.drop("Unnamed: 0", axis=1, inplace=True)
testData.drop("Unnamed: 0", axis=1, inplace=True)

trainingData.to_csv("trainingData",index=False)
testData.to_csv("testData",index=False)"""

#Created trainingData by trainingData csv file which created frome above
trainingData = pd.read_csv("trainingData.csv")
testData = pd.read_csv("testData.csv")


size = len(testData)
sizeTrue = len(testData.loc[testData["label"]==1])
tempDfJaccard = testData.loc[testData["jaccard"]>0.5]


print(tempDfJaccard.head())

print(testData.sort_values(by=['jaccard'],ascending=False).head(2))
print()

print(testData.sort_values(by=['adamic'],ascending=False).head(2))
print()

print(testData.sort_values(by=['pref_at'],ascending=False).head(2))
print()

print(testData.sort_values(by=['common_n'],ascending=False).head(2))
print()

jaccard_pred_results_with0 = (len(tempDfJaccard)/(size))*100
jaccard_pred_results_only1 = (len(tempDfJaccard)/(sizeTrue))*100
print("jaccard prediction with 0's result is:\t", jaccard_pred_results_with0)
print("jaccard prediction only label1 result is:\t", jaccard_pred_results_only1)
print()



trainingData["label"] = trainingData["label"].astype("category")

trainingData = trainingData.sample(n=len(trainingData),random_state=49)
testData = testData.sample(n=len(testData),random_state=49)

columns = ["jaccard", "adamic", "pref_at", "common_n"]
X = trainingData[columns]
y = trainingData["label"]


#rf_random.fit(X, y)

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def evaluate_model(predictions, actual):
    accuracy = accuracy_score(actual, predictions)
    precision = precision_score(actual, predictions)
    recall = recall_score(actual, predictions)
    f1Score = f1_score(actual,predictions)

    metrics = ["accuracy", "precision", "recall","f1_score"]
    values = [accuracy, precision, recall,f1Score]
    return pd.DataFrame(data={'metric': metrics, 'value': values})


def feature_importance(columns, classifier):
    features = list(zip(columns, classifier.feature_importances_))

    sorted_features = sorted(features, key=lambda x: x[1] * -1)

    keys = [value[0] for value in sorted_features]
    values = [value[1] for value in sorted_features]
    return pd.DataFrame(data={'feature': keys, 'value': values})

"""n_estimators = [10,16,32,50,64,100]
max_depth = [5,10,16,32]"""


#classifier = RandomForestClassifier(n_estimators=15, max_depth=5, random_state=0)
#outFile = open('output.txt','w+')


"""for i in n_estimators:
    for j in max_depth:
        classifier = RandomForestClassifier(n_estimators=i, max_depth=j,
                                    random_state=0)
        classifier.fit(X, y)
        print(f"number of tree = {i}\tnumber of max_depth = {j}")
        print(f"number of tree = {i}\tnumber of max_depth = {j}",file=outFile)
        predictions = classifier.predict(testData[columns])
        y_test = testData["label"]
        print(evaluate_model(predictions, y_test))
        print(evaluate_model(predictions, y_test),file=outFile)
        print()
        print(file=outFile)
        print(classification_report(y_test, predictions))
        print(classification_report(y_test, predictions),file=outFile)
        print()
        print(file=outFile)
        print(feature_importance(columns, classifier))
        print(feature_importance(columns, classifier),file=outFile)
        print()
        print(file=outFile)
"""
classifier = RandomForestClassifier(n_estimators=32, max_depth=16,
                                    random_state=0)
classifier.fit(X, y)
print("Random Forest Example Result:")
print(f"number of tree = {32}\tnumber of max_depth = {16}")

predictions = classifier.predict(testData[columns])
y_test = testData["label"]

print(evaluate_model(predictions, y_test))

print(feature_importance(columns, classifier))

"""
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=32,max_depth=16,random_state=0)
print()
print("Gradient Boosting Example Result:")
clf.fit(X,y)
predictionsClf = clf.predict(testData[columns])
y_testG = testData["label"]

print(evaluate_model(predictionsClf, y_testG))

print(feature_importance(columns, clf))
print()

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=32,max_depth=16,random_state=0)
print()
print("Ada Boosting Example Result:")
ada.fit(X,y)
predictionsAda = ada.predict(testData[columns])
y_testA = testData["label"]

print(evaluate_model(predictionsAda, y_testA))

print(feature_importance(columns, ada))
print()"""
