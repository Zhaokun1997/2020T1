import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    # load data
    data_frame = pd.read_csv('house_prices.csv')
    feature_vectors = ['house age', 'distance to the nearest MRT station', 'number of convenience stores',
                       'house price of unit area']
    # step 1: pre-processing data
    # 3 features
    data_house_age = data_frame[feature_vectors[0]]
    data_distance_to_station = data_frame[feature_vectors[1]]
    data_nb_of_stores = data_frame[feature_vectors[2]]

    # 1 output
    data_house_price = data_frame[feature_vectors[-1]]

    # get normalized data
    max_house_age = max(list(data_house_age))
    min_house_age = min(list(data_house_age))
    new_house_age = [((x - min_house_age) / (max_house_age - min_house_age))
                     for x in list(data_house_age)]

    max_distance_to_station = max(list(data_distance_to_station))
    min_distance_to_station = min(list(data_distance_to_station))
    new_distance_to_station = [((x - min_distance_to_station) / (max_distance_to_station - min_distance_to_station))
                               for x in list(data_distance_to_station)]

    max_nb_of_stores = max(list(data_nb_of_stores))
    min_nb_of_stores = min(list(data_nb_of_stores))
    new_nb_of_stores = [((x - min_nb_of_stores) / (max_nb_of_stores - min_nb_of_stores))
                        for x in list(data_nb_of_stores)]

    # step 2: Creating test and training set
    # total data y
    data_house_price_y = np.array(data_house_price).reshape(-1, 1)
    # total data x for three features
    data_house_age_x = np.array(new_house_age).reshape(-1, 1)
    data_distance_to_station_x = np.array(new_distance_to_station).reshape(-1, 1)
    data_nb_of_stores_x = np.array(new_nb_of_stores).reshape(-1, 1)
    # split data to training data and test data
    # shuffle == false <==> not random
    train_data_house_age_x, test_data_house_age_x, train_data_house_price_y, test_data_house_price_y = \
        train_test_split(data_house_age_x, data_house_price_y, test_size=0.25, shuffle=False)

    # step 3: Stochastic gradient descent
    loss_record = []  # record loss function value
    iter_record = []  # record iteration value
    training_len = len(train_data_house_age_x)
    max_iteration = 50
    alpha = 0.01  # learning rate
    theta = [-1, -0.5]  # 𝜃 coefficients
    iter_count = 0
    error = 0

    while iter_count < max_iteration:
        loss = 0
        i = random.randint(0, training_len - 1)  # choose an observation in training dataset randomly
        predicted_fuc = theta[0] * 1 + theta[1] * train_data_house_age_x[i][0]
        theta[0] = theta[0] + alpha * (train_data_house_price_y[i][0] - predicted_fuc) * 1
        theta[1] = theta[1] + alpha * (train_data_house_price_y[i][0] - predicted_fuc) * train_data_house_age_x[i][0]
        for j in range(training_len):
            predicted_fuc = theta[0] * 1 + theta[1] * train_data_house_age_x[j][0]
            error = (train_data_house_price_y[j][0] - predicted_fuc) ** 2
            loss = loss + error
        loss = (1 / training_len) * loss
        iter_count += 1
        loss_record.append(loss)
        iter_record.append(iter_count)

    # step 4: Visualization
    iter_x = np.array(iter_record).reshape(-1, 1)
    loss_y = np.array(loss_record).reshape(-1, 1)
    # model = linear_model.LinearRegression()
    # model.fit(iter_x, loss_y)
    # predicted_y = model.predict(iter_x)
    plt.scatter(iter_x, loss_y, color="red")
    plt.show()

    # step 5: Evaluation


