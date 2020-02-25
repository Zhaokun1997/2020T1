import requests
import pandas as pd


def get_json(url):
    """
    :param url: URL of the resource
    :return: JSON
    """
    resp = requests.get(url=url)
    data = resp.json()
    return data


def json_to_dataframe(json_obj):
    """
    The root element contains two main elements : data and meta;
    data: contains the statistics for a given zip code, and
    meta: contains the information about the columns
    :param json_obj: JSON object for the dataset
    :return: a dataframe
    """
    # let's get the list of statistics for all zip codes
    json_data = json_obj['data']

    # to create a dataframe we also need the name of the columns:
    columns = []
    for c in json_obj['meta']['view']['columns']:
        columns.append(c['name'])

    return pd.DataFrame(data=json_data, columns=columns)


def print_dataframe(data_frame, print_columns=True, print_rows=True):
    # print column names
    if print_columns:
        print(",".join([column for column in data_frame]))

    # print rows one by one
    if print_rows:
        for index, row in data_frame.iterrows():
            print(",".join([str(row[column]) for column in data_frame]))


if __name__ == '__main__':
    url = "https://data.cityofnewyork.us/api/views/kku6-nxdu/rows.json"

    print("Start to fetch json...")
    json_obj = get_json(url)
    print("fetch json source successfully...")

    print("Convert the json object to a dataframe...")
    data_frame = json_to_dataframe(json_obj)
    print("Convert the json object to a dataframe successfully...")

    print_dataframe(data_frame, True, True)
