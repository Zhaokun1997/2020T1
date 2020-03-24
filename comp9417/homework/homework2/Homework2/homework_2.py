import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import random
import math
import copy


# homework_2
# Question 2

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

    # split dataset
    training_dataset = df.loc[0:620, :]
    testing_dataset = df.loc[620:887, :]

    # training data and class labels
    training_dataset_x = training_dataset.loc[:, feature_vectors]
    testing_dataset_x = testing_dataset.loc[:, feature_vectors]
    training_dataset_y = training_dataset.loc[:, ['Survived']]
    testing_dataset_y = testing_dataset.loc[:, ['Survived']]

    # PART A:
    # create DT model and train model
    clf = DecisionTreeClassifier()
    clf.fit(training_dataset_x, training_dataset_y)

    # for training data
    training_predicted = clf.predict(training_dataset_x)
    print("accuracy score for training data: ", accuracy_score(training_dataset_y, training_predicted))
    # for testing data
    testing_predicted = clf.predict(testing_dataset_x)
    print("accuracy score for testing data: ", accuracy_score(testing_dataset_y, testing_predicted))
    # print("accuracy score for training data: ", clf.score(training_dataset_x, training_dataset_y))
    # print("accuracy score for training data: ", clf.score(testing_dataset_x, testing_dataset_y))

    # PART B & C:
    iter_list = []
    auc_train = []  # store train AUC values
    auc_test = []  # store test AUC values
    for i in range(2, 21, 1):
        temp_clf = DecisionTreeClassifier(min_samples_leaf=i)
        temp_clf.fit(training_dataset_x, training_dataset_y)
        temp_train_predicted = temp_clf.predict(training_dataset_x)
        temp_test_predicted = temp_clf.predict(testing_dataset_x)
        auc_train_value = roc_auc_score(training_dataset_y, temp_train_predicted)
        auc_test_value = roc_auc_score(testing_dataset_y, temp_test_predicted)
        iter_list.append(int(i))
        auc_test.append(auc_test_value)
        auc_train.append(auc_train_value)

    # B:
    idx_maxValue = auc_test.index(max(auc_test))
    optimal_nb = iter_list[idx_maxValue]
    print("optimal number of min_samples_leaf: ", optimal_nb)

    # C:
    # for training data
    plt.plot(iter_list, auc_train, label="train_AUC_value", color='green')
    plt.xlabel("iteration")
    plt.ylabel("auc score")
    plt.xticks(iter_list)  # show x-coordinate with details
    plt.legend()  # show label graphic
    plt.show()

    # for testing data
    plt.plot(iter_list, auc_test, label="test_AUC_value", color='red')
    plt.xlabel("iteration")
    plt.ylabel("auc score")
    plt.xticks(iter_list)  # show x-coordinate with details
    plt.legend()  # show label graphic
    plt.show()

    # PART D:
    total = df[(df['Sex'] == 1.0) & (df['Pclass'] == 0.0)]
    survived = total[total['Survived'] == 1]
    posterior_p = len(survived) / len(total)
    print("posterior probability that part D: ", posterior_p)
