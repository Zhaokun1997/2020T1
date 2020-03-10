import ast
import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import copy
from collections import defaultdict

studentid = os.path.basename(sys.modules[__name__].__file__)


#################################################
# Your personal methods can be here ...
def clean_cast(x):
    origin_list = ast.literal_eval(x)  # parsing original string: '[{}]' --> [{}]
    character_list = [item['character'] for item in origin_list]  # get movie characters' names
    result = sorted(character_list)
    returnValue = ','.join(result)
    return returnValue


#################################################


def log(question, output_df, other):
    print("--------------- {}----------------".format(question))
    if other is not None:
        print(question, other)
    if output_df is not None:
        print(output_df.head(5).to_string())


def question_1(movies, credits):
    """
    :param movies: the path for the movie.csv file
    :param credits: the path for the credits.csv file
    :return: df1
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    df_credits = pd.read_csv(credits)
    df_movies = pd.read_csv(movies)

    # 'how = inner' indicates only the intersection part of two dataframes will be kept
    df1 = pd.merge(df_movies, df_credits, how='inner', on=['id'])
    #################################################

    log("QUESTION 1", output_df=df1, other=df1.shape)
    return df1


def question_2(df1):
    """
    :param df1: the dataframe created in question 1
    :return: df2
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    my_columns = ['id', 'title', 'popularity', 'cast', 'crew',
                  'budget', 'genres', 'original_language', 'production_companies',
                  'production_countries', 'release_date', 'revenue', 'runtime',
                  'spoken_languages', 'vote_average', 'vote_count']
    total_columns = list(df1.columns)
    drop_columns = [column for column in total_columns if column not in my_columns]
    df1.drop(drop_columns, axis=1, inplace=True)
    df2 = df1
    #################################################

    log("QUESTION 2", output_df=df2, other=(len(df2.columns), sorted(df2.columns)))
    return df2


def question_3(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df3
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    df2 = df2.set_index('id')
    df3 = df2
    #################################################

    log("QUESTION 3", output_df=df3, other=df3.index.name)
    return df3


def question_4(df3):
    """
    :param df3: the dataframe created in question 3
    :return: df4
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    df3.drop(df3[df3.budget == 0].index, inplace=True)
    df4 = df3
    #################################################

    log("QUESTION 4", output_df=df4, other=(df4['budget'].min(), df4['budget'].max(), df4['budget'].mean()))
    return df4


def question_5(df4):
    """
    :param df4: the dataframe created in question 4
    :return: df5
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    df4['success_impact'] = df4.apply(lambda x: (x.revenue - x.budget) / x.budget, axis=1)
    df5 = df4
    #################################################

    log("QUESTION 5", output_df=df5,
        other=(df5['success_impact'].min(), df5['success_impact'].max(), df5['success_impact'].mean()))
    return df5


def question_6(df5):
    """
    :param df5: the dataframe created in question 5
    :return: df6
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    min_pop = df5['popularity'].min()
    max_pop = df5['popularity'].max()
    df5['popularity'] = df5.apply(lambda x: 100 * float((x.popularity - min_pop) / (max_pop - min_pop)), axis=1)
    df6 = df5
    #################################################

    log("QUESTION 6", output_df=df6, other=(df6['popularity'].min(), df6['popularity'].max(), df6['popularity'].mean()))
    return df6


def question_7(df6):
    """
    :param df6: the dataframe created in question 6
    :return: df7
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    df6['popularity'] = df6['popularity'].astype('int16')
    df7 = df6
    #################################################

    log("QUESTION 7", output_df=df7, other=df7['popularity'].dtype)
    return df7


def question_8(df7):
    """
    :param df7: the dataframe created in question 7
    :return: df8
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    df7['cast'] = df7['cast'].apply(clean_cast)
    df8 = df7
    #################################################

    log("QUESTION 8", output_df=df8, other=df8["cast"].head(10).values)
    return df8


def question_9(df8):
    """
    :param df9: the dataframe created in question 8
    :return: movies
            Data Type: List of strings (movie titles)
            Please read the assignment specs to know how to create the output
    """

    #################################################
    # Your code goes here ...
    df9 = copy.deepcopy(df8)
    df9['cast'] = df9['cast'].apply(lambda x: len(str(x).split(",")))
    df9.sort_values(by=['cast'], ascending=False, inplace=True)
    movies = list(df9['title'].head(10).values)
    #################################################

    log("QUESTION 9", output_df=None, other=movies)
    return movies


def question_10(df8):
    """
    :param df8: the dataframe created in question 8
    :return: df10
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    df10 = copy.deepcopy(df8)
    df10['release_date'] = pd.to_datetime(df10['release_date'])
    df10.sort_values(by=['release_date'], ascending=False, inplace=True)
    #################################################

    log("QUESTION 10", output_df=df10, other=df10["release_date"].head(5).to_string().replace("\n", " "))
    return df10


def question_11(df10):
    """
    :param df10: the dataframe created in question 10
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # Your code goes here ...
    count_genres = defaultdict(int)
    for grs in df10['genres']:
        origin_list = ast.literal_eval(grs)  # "[{'id': 80, 'name': 'Crime'}, {'id': 18, 'name': 'Drama'}]" --> []
        name_list = [item['name'] for item in origin_list]  # get all names for every row
        for name in name_list:
            count_genres[name] += 1

    # merge process
    sort_list = sorted(count_genres.items(), key=lambda x: x[1])  # sorted by values: return a list
    merge = [sort_list[i][0] for i in range(4)]  # get the most infrequence 4 genres
    count_genres['others'] = sum([count_genres[name] for name in merge])  # merge 4 genres to others
    for name in merge:
        del count_genres[name]

    keys = list(count_genres.keys())
    values = list(count_genres.values())
    res_df = pd.Series(values, index=keys, name='figure: question 11')  # create Series
    res_df.plot.pie(autopct='%.2f%%', figsize=(9, 9))
    plt.show()
    #################################################
    plt.savefig("{}-Q11.png".format(studentid))


def question_12(df10):
    """
    :param df10: the dataframe created in question 10
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # Your code goes here ...
    count_countries = defaultdict(int)
    for record in df10['production_countries']:
        origin_list = ast.literal_eval(record)
        for dct in origin_list:
            count_countries[dct['name']] += 1
    # sort the dictionary by key with alphabetical order
    # return a list with tuple [ (, ), (, ) ]
    res_list = sorted(count_countries.items(), key=lambda x: x[0])
    keys = [t[0] for t in res_list]
    values = [t[1] for t in res_list]

    res_df = pd.Series(values, index=keys, name='q12')
    res_df.plot.bar()
    plt.show()
    #################################################

    plt.savefig("{}-Q12.png".format(studentid))


def question_13(df10):
    """
    :param df10: the dataframe created in question 10
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # Your code goes here ...

    #################################################

    plt.savefig("{}-Q13.png".format(studentid))


if __name__ == "__main__":
    df1 = question_1("movies.csv", "credits.csv")
    df2 = question_2(df1)
    df3 = question_3(df2)
    df4 = question_4(df3)
    df5 = question_5(df4)
    df6 = question_6(df5)
    df7 = question_7(df6)
    df8 = question_8(df7)
    movies = question_9(df8)
    df10 = question_10(df8)
    question_11(df10)
    question_12(df10)
    # question_13(df10)
