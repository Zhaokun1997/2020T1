import json
import pandas as pd
from pymongo import MongoClient


def read_csv(csv_file):
    """
    :param csv_file: the path of csv file
    :return: a dataframe out of the csv file
    """
    return pd.read_csv(csv_file)


def print_data_frame(data_frame, print_columns=True, print_rows=True):
    # print column attributes/names:
    if print_columns:
        print(",".join([column for column in data_frame]))
    if print_rows:
        for index, row in data_frame.iterrows():
            print(",".join([str(row[column]) for column in data_frame]))


def write_in_mongodb(data_frame, mongo_host, mongo_port, db_name, collection):
    """
    :param data_frame
    :param mongo_host: mongoDB server address
    :param mongo_port: mongoDB server port number
    :param db_name: the name of the database
    :param collection: the name of the collection inside the database
    """
    client = MongoClient(host=mongo_host, port=mongo_port)
    db = client[db_name]
    cnx = db[collection]

    # you can only store documents in mongodb;
    # so you need to convert rows inside the dataframe into a list of json objects
    records = json.loads(data_frame.T.to_json()).values()
    cnx.insert_many(records)
    # cnx.insert(records)  # is deprecated


def read_from_mongodb(mongo_host, mongo_port, db_name, collection):
    """
    :param mongo_host: mongoDB server address
    :param mongo_port: mongoDB server port number
    :param db_name: the name of the database
    :param collection: the name of the collection inside the database
    """
    client = MongoClient(host=mongo_host, port=mongo_port)
    db = client[db_name]
    cnx = db[collection]
    return pd.DataFrame(list(cnx.find()))


if __name__ == '__main__':
    db_name = 'comp9321'
    mongo_port = 27017
    mongo_host = 'localhost'

    # obtain dataframe from csv file
    print("Read data from csv file...")
    csv_file = 'Demographic_Statistics_By_Zip_Code.csv'
    data_frame = read_csv(csv_file)
    print("Read data from csv file successfully...")

    # write data(obtained from csv file) to the mongoDB server
    collection = 'Demographic_Statistics'
    print("Writing into the mongodb...")
    write_in_mongodb(data_frame, mongo_host, mongo_port, db_name, collection)
    print("Writing into the mongodb successfully...")

    # query data from mongoDB server
    print("Querying the mongo database...")
    data_frame = read_from_mongodb(mongo_host, mongo_port, db_name, collection)
    print_data_frame(data_frame, True, True)
    print("Querying the mongo database successfully...")
