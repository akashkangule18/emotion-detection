import pandas as pd
import numpy as np
import os

import yaml
from sklearn.model_selection import train_test_split

# making seperate functions

def param(file_path):
    test_size  = yaml.safe_load(open(file_path,'r'))['data_ingestion']['test_size']
    return test_size

def load_data(url):
    df = pd.read_csv(url)
    return df

def clean_(df):
    final_df = df[df['sentiment'].isin(['happiness','sadness'])]
    final_df.drop(columns='tweet_id',inplace=True)
    final_df['sentiment'] = final_df['sentiment'].replace({'happiness':1, 'sadness':0})

    return final_df



def save_data(data_path, training_data, testing_data):

    os.makedirs(data_path, exist_ok= True)
    # saving trainig and testing data into csv files
    training_data.to_csv(os.path.join(data_path, 'train.csv'), index = False)
    testing_data.to_csv(os.path.join(data_path, 'test.csv'), index = False)


def main():

    test_size = param('params.yaml')
    df = load_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv' )
    final_df = clean_(df)
    training_data, testing_data = train_test_split(final_df, test_size= test_size, random_state= 42)
    
          # creating data_path
    data_path = os.path.join('data','raw')

    save_data(data_path, training_data, testing_data)



if __name__ == "__main__":
    main()
    













