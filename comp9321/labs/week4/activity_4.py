import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    csv_file = 'iris.csv'
    df = pd.read_csv(csv_file)

    # divide the dataset into three dataframes based on the species
    setosa_df = df.query('species == "setosa"')
    versicolor_df = df.query('species == "versicolor"')
    virginica_df = df.query('species == "virginica"')

    # create subplots
    fig, axes = plt.subplots(nrows=1, ncols=2)

    # Plot a scatter chart using x='sepal_length', y='sepal_width', and separate colors for each of the three dataframes
    # "ax=ax1" specify which subplot we should draw on
    ax = setosa_df.plot.scatter(x='sepal_length', y='sepal_width', label='setosa', c='blue', ax=axes[0])
    versicolor_df.plot.scatter(x='sepal_length', y='sepal_width', label='versicolor', c='red', ax=ax)
    virginica_df.plot.scatter(x='sepal_length', y='sepal_width', label='virginica_df', c='green', ax=ax)

    # Plot a scatter chart using x='petal_length', y='petal_width', and separate colors for each of the three dataframes
    ax = setosa_df.plot.scatter(x='petal_length', y='petal_width', label='setosa', c='blue', ax=axes[1])
    versicolor_df.plot.scatter(x='petal_length', y='petal_width', label='versicolor', c='red', ax=ax)
    virginica_df.plot.scatter(x='petal_length', y='petal_width', label='virginica_df', c='green', ax=ax)

    plt.show()
