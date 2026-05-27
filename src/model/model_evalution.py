import pandas as pd
import numpy as np
import os
import json
import pickle

# fetching testing data from the transformerd
def data(test_path):

    testing_data = pd.read_csv(test_path)

    X_test = testing_data.iloc[:,0:-1].values
    y_test = testing_data.iloc[:,-1].values
    return X_test, y_test




def score(y_test, y_pred, y_proba):
    # getting accuracy metrics 
    from sklearn.metrics import accuracy_score, recall_score, precision_score,roc_auc_score

    accuracy = accuracy_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    auc = roc_auc_score(y_test,y_proba)

    return accuracy, recall, precision, auc

def metrics(accuracy, recall, precision, auc):
    # defining the metrics
    metrics_dict = {
        'accuracy' : accuracy,
        'recall' : recall,
        'precision' : precision,
        'auc':auc
    }

    # getting json file
    with open('reports/metrics.json','w') as file:
        json.dump(metrics_dict, file, indent=4)

def main():
        X_test,y_test = data('./data/processed/test.csv')
        clf = pickle.load(open('models/model.pkl','rb'))
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:,1]
        accuracy, recall, precision, auc = score(y_test, y_pred, y_proba)
        metrics(accuracy, recall, precision, auc)


if __name__ == '__main__':
     main()

