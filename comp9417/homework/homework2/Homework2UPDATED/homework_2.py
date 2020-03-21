import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import math


# data normalisation
def pre_processing(dataframe, features):
    for ft in features:
        max_value = dataframe[ft].max()
        min_value = dataframe[ft].min()
        dataframe[ft] = [(x - min_value) / (max_value - min_value) for x in dataframe[ft]]
    return dataframe


if __name__ == '__main__':
    # load data
    csv_file = 'titanic.csv'
    df = pd.read_csv(csv_file)
    feature_vectors = ['Pclass', 'Sex', 'Age', 'Siblings_Spouses_Aboard', 'Parents_Children_Aboard']

    # pre-processing data
    df = pre_processing(df, feature_vectors)

    # split training and testing sets
    training_dataset = df.loc[0:620, :]
    testing_dataset = df.loc[620:887, :]

    DecisionTreeClassifier()

