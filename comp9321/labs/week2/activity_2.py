import sqlite3
import pandas as pd
from pandas.io import sql


def read_csv(csv_file):
    """
    :param csv_file: the path of csv file
    :return: a dataframe out of the csv file
    """
    return pd.read_csv(csv_file)


def write_in_sqlite(data_frame, database_file, table_name):
    """
    :param data_frame: the dataframe which must be written into the database
    :param database_file: where the database is stored
    :param table_name: the name of the table
    """
    cnx = sqlite3.connect(database_file) # make a connection
    sql.to_sql(data_frame, name=table_name, con=cnx)


def read_from_sqlite(database_file, table_name):
    """
    :param database_file: where the database is stored
    :param table_name: the name of the table
    :return: a dataframe
    """
    cnx = sqlite3.connect(database_file)
    return sql.read_sql('select * from ' + table_name, cnx)


if __name__ == '__main__':
    table_name = "Demographic_Statistics"
    database_file = 'Demographic_Statistics.db'  # name of sqlite db file that will be created
    csv_file = 'Demographic_Statistics_By_Zip_Code.csv'

    # read data from csv file
    print("Reading csv file...")
    loaded_df = read_csv(csv_file)
    print("Obtain data from csv file successfully...")

    # write data (read from csv file) to the sqlite database
    print("Creating database...")
    write_in_sqlite(loaded_df, database_file, table_name)
    print("Write data to database successfully...")

    # make a query
    print("Querying the database")
    queried_df = read_from_sqlite(database_file, table_name)
