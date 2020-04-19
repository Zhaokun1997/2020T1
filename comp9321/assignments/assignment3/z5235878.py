import pandas as pd
import numpy as np
import sys
import ast
import json
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
from sklearn.metrics import *
from decimal import Decimal


def load_data(path1, path2):
    df_train_origin = pd.read_csv(path1)
    df_validation_origin = pd.read_csv(path2)

    # shuffle all data
    df_train_origin = shuffle(df_train_origin)
    df_validation_origin = shuffle(df_validation_origin)
    if df_train_origin is not None and df_validation_origin is not None:
        return df_train_origin, df_validation_origin
    else:
        print("Failed to load datasets")
        sys.exit()


def feature_engineering(df_train_origin, df_validation_origin):
    feature_columns = ["cast", "crew", "budget", "original_language", "genres", "release_date", "runtime"]
    df_train_X = df_train_origin[feature_columns]  # part_1 training data X
    df_validation_X = df_validation_origin[feature_columns]  # part_1 validation data X
    return df_train_X, df_validation_X


def data_pre_processing(df_train_X, df_validation_X):
    # regulazition all data sets

    # step 1: extract cast lead name in cast
    # for training data
    lead_star_list_train = []  # a list stores all leading stars
    for cast in df_train_X['cast']:
        cast = ast.literal_eval(cast)
        lead_star_list_train.append(cast[0]['name'])
    df_train_X['cast'] = lead_star_list_train

    # for testing data
    lead_star_list_validation = []  # a list stores all leading stars
    for cast in df_validation_X['cast']:
        cast = ast.literal_eval(cast)
        lead_star_list_validation.append(cast[0]['name'])
    df_validation_X['cast'] = lead_star_list_validation

    # step 2: extract director name in crew
    # for training data
    director_list_train = []  # a list stores all leading stars
    for crew in df_train_X['crew']:
        crew = ast.literal_eval(crew)
        for member in crew:
            if member['job'] == "Director":
                director_list_train.append(member['name'])
                break
    df_train_X['crew'] = director_list_train

    # for testing data
    director_list_validation = []  # a list stores all leading stars
    for crew in df_validation_X['crew']:
        crew = ast.literal_eval(crew)
        for member in crew:
            if member['job'] == "Director":
                director_list_validation.append(member['name'])
                break
    df_validation_X['crew'] = director_list_validation

    # step 3: extract main genres
    # for training data
    genres_list_train = []
    for pc in df_train_X['genres']:
        pc = ast.literal_eval(pc)
        genres_list_train.append(pc[0]['id'])
    df_train_X['genres'] = genres_list_train
    # for testing data
    genres_list_validation = []
    for pc in df_validation_X['genres']:
        pc = ast.literal_eval(pc)
        genres_list_validation.append(pc[0]['id'])
    df_validation_X['genres'] = genres_list_validation

    # step 4: extract all months
    # for training data
    month_train = []
    for date in df_train_X['release_date']:
        month_train.append(int(date[5:7]))
    df_train_X['release_date'] = month_train

    # for testing data
    month_validation = []
    for date in df_validation_X['release_date']:
        month_validation.append(int(date[5:7]))
    df_validation_X['release_date'] = month_validation

    return df_train_X, df_validation_X


def data_encoding(df_train_X, df_validation_X):
    # encoding data
    cast_set = set(df_train_X['cast']).union(set(df_validation_X['cast']))
    cast_dict = dict()
    cast_list = sorted(list(cast_set))
    index = 1
    for name in cast_list:
        cast_dict[name] = index
        index += 1
    # convert name to id
    for i in range(len(df_train_X)):
        name = df_train_X.loc[i, 'cast']
        df_train_X.loc[i, 'cast'] = cast_dict[name]
    for i in range(len(df_validation_X)):
        name = df_validation_X.loc[i, 'cast']
        df_validation_X.loc[i, 'cast'] = cast_dict[name]

    crew_set = set(df_train_X['crew']).union(set(df_validation_X['crew']))
    crew_dict = dict()
    crew_list = sorted(list(crew_set))
    index = 1
    for name in crew_list:
        crew_dict[name] = index
        index += 1
    # convert name to id
    for i in range(len(df_train_X)):
        name = df_train_X.loc[i, 'crew']
        df_train_X.loc[i, 'crew'] = crew_dict[name]
    for i in range(len(df_validation_X)):
        name = df_validation_X.loc[i, 'crew']
        df_validation_X.loc[i, 'crew'] = crew_dict[name]

    language_set = set(df_train_X['original_language']).union(set(df_validation_X['original_language']))
    language_dict = dict()
    language_list = sorted(list(language_set))
    index = 1
    for name in language_list:
        language_dict[name] = index
        index += 1

    # convert name to id
    for i in range(len(df_train_X)):
        name = df_train_X.loc[i, 'original_language']
        df_train_X.loc[i, 'original_language'] = language_dict[name]
    for i in range(len(df_validation_X)):
        name = df_validation_X.loc[i, 'original_language']
        df_validation_X.loc[i, 'original_language'] = language_dict[name]
    return df_train_X, df_validation_X


if __name__ == '__main__':
    # $ python3 z{id}.py path1 path2
    if len(sys.argv) != 3:
        print("Please give all arguments in right form!")
        print("The right form is: $ python3 z{id}.py path1 path2")
        sys.exit()
    path1, path2 = sys.argv[1], sys.argv[2]
    df_train_origin, df_validation_origin = load_data(path1, path2)
    df_train_X, df_validation_X = feature_engineering(df_train_origin, df_validation_origin)
    df_train_X, df_validation_X = data_pre_processing(df_train_X, df_validation_X)
    df_train_X, df_validation_X = data_encoding(df_train_X, df_validation_X)

    # part1 model
    model1 = LinearRegression()
    model1.fit(df_train_X, df_train_origin['revenue'])
    predicted_y_part1 = model1.predict(df_validation_X)

    # generate summary.csv for part1
    MSR = mean_squared_error(df_validation_origin['revenue'], predicted_y_part1)
    correlation = round(pearsonr(df_validation_origin['revenue'], predicted_y_part1)[0],
                        2)  # a tuple (correlaction, R-value)

    df_part1_summary = pd.DataFrame()
    df_part1_summary['zid'] = ["z5235878"]
    df_part1_summary['MSR'] = [MSR]
    df_part1_summary['correlation'] = [correlation]
    df_part1_summary.to_csv("z5235878.PART1.summary.csv")

    # generate output.csv for part1
    movie_id = df_validation_origin['movie_id']
    predicted_revenue = predicted_y_part1
    df_part1_output = pd.DataFrame()
    df_part1_output['movie_id'] = movie_id
    df_part1_output['predicted_revenue'] = predicted_revenue
    df_part1_output.to_csv("z5235878.PART1.output.csv")

    # part2 model
    # do cross-validation
    k_range = range(1, 31, 2)
    k_scores = []

    # iterate different k to determine the best k with the best perfromance
    # using cross validation
    for k in k_range:
        model2 = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model2, df_train_X, df_train_origin['rating'], cv=10, scoring='accuracy')
        k_scores.append(scores.mean())

    # plot
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()

    # train model
    best_k = k_range[k_scores.index(max(k_scores))]
    print("best k we use for KNN Classifier is: ", best_k)
    final_k = 19
    model2 = KNeighborsClassifier(n_neighbors=final_k)
    model2.fit(df_train_X, df_train_origin['rating'])
    predicted_y_part2 = model2.predict(df_validation_X)

    # generate summary.csv for part2
    # Decimal(a).quantize(Decimal("0.00"))
    average_precision = Decimal(
        round(precision_score(df_validation_origin['rating'], predicted_y_part2, average="macro"), 3)).quantize(
        Decimal("0.00"))
    average_recall = Decimal(
        round(recall_score(df_validation_origin['rating'], predicted_y_part2, average="weighted"), 2)).quantize(
        Decimal("0.00"))
    accuracy = Decimal(round(accuracy_score(df_validation_origin['rating'], predicted_y_part2), 2)).quantize(
        Decimal("0.00"))
    df_part2_summary = pd.DataFrame()
    df_part2_summary['average_precision'] = [average_precision]
    df_part2_summary['average_recall'] = [average_recall]
    df_part2_summary['accuracy'] = [accuracy]
    df_part2_summary.to_csv("z5235878.PART2.summary.csv")

    # generate output.csv for part1
    predicted_rating = predicted_y_part2
    df_part2_output = pd.DataFrame()
    df_part2_output['movie_id'] = movie_id
    df_part2_output['predicted_rating'] = predicted_rating
    df_part2_output.to_csv("z5235878.PART2.output.csv")

    # output
    # part 1
    print("++++++++++++ part 1 +++++++++++++")
    print("MSE: ", MSR)
    print("correlation: ", correlation)

    # part 2
    print("++++++++++++ part 2 +++++++++++++")
    print("precision_score: ", average_precision)
    print("recall_score: ", average_recall)
    print("accuracy_score: ", accuracy)
