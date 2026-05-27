import pandas as pd
import numpy as np
import os
import pickle
import yaml
from logger import logger

def load_file(file_path):
    try:
        params = yaml.safe_load(open(file_path,'r'))['model_building']
        logger.info('params are fetched succesfully')
        return params
    except Exception as e:
        logger.error(f"params are not fetched please check the file path")


# getiing the transformed data
def data(train_path):
    try: 
        training_data = pd.read_csv(train_path)

        # seperating trainign and testing data
        X_train = training_data.iloc[:,0:-1]
        y_train = training_data.iloc[:,-1]
        logger.info('trained data has been fetched sucessfully')
        return X_train, y_train
    except Exception as e:
        logger.error(f"training data is not fetched succesfully, please check the path of training data")



def model(X_train, y_train,n_estimators, max_samples):

    # building model
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators= n_estimators, max_samples= max_samples)
    clf = clf.fit(X_train,y_train)
    return clf


# saving the model


def main():
    params = load_file('params.yaml')
    X_train, y_train = data('./data/processed/train.csv')
    clf = model(X_train, y_train, params['n_estimators'],params['max_samples'])
    pickle.dump(clf,open('models/model.pkl','wb'))

if __name__ == '__main__':
    main()



