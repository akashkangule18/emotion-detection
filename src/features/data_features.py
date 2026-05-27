import pandas as pd
import numpy as np
import os
import yaml


def load_file(file_path):
    max_features = yaml.safe_load(open(file_path,'r'))['data_features']['max_features']
    return max_features

def data(train_path, test_path):

    # getting the preprocessed data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    return train_data, test_data

def transform(train_data, test_data, max_features):

    # apply transformation code on prerocessed data
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features= max_features)

    # filling null values
    train_data.fillna('', inplace= True)
    test_data.fillna('', inplace= True)

    # seperating data at first 
    X_train  = train_data['content'].values
    X_test = test_data['content'].values
    y_train = train_data['sentiment'].values
    y_test = test_data['sentiment'].values

    # Fit the vectorizer on the training data and transform it
    X_train_bow = vectorizer.fit_transform(X_train)

    # Transform the test data using the same vectorizer
    X_test_bow = vectorizer.transform(X_test)

    # again merging  the whole data
    train_df = pd.DataFrame(
        X_train_bow.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    test_df = pd.DataFrame(
        X_test_bow.toarray(),
        columns= vectorizer.get_feature_names_out()
    )

    # adding output column in data
    train_df['sentiment'] = y_train
    test_df['sentiment'] = y_test

    return train_df, test_df


def save_data(data_path,train_df, test_df):
    os.makedirs(data_path, exist_ok= True)

    train_df.to_csv(os.path.join(data_path,'train.csv'), index = False)
    test_df.to_csv(os.path.join(data_path,'test.csv'), index = False)



def main_():
    max_features = load_file('params.yaml')
    train_data, test_data = data('./data/interim/train_processed.csv','./data/interim/test_processed.csv')
    train_df , test_df =  transform(train_data, test_data, max_features)
    data_path = os.path.join('data','processed')
    save_data(data_path,train_df, test_df)

if __name__ == '__main__':
    main_()

