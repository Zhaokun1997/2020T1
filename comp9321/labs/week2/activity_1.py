import pandas as pd


def read_csv(csv_file):
    """
    :para csv_file: find the path of csv file
    :return: a dataframe out of the csv file
    """
    return pd.read_csv(csv_file)


def write_in_csv(data_frame, dest_csv_file):
    """
    :para csv_file: find the path of csv file
    :return: a dataframe out of the csv file
    """
    data_frame.to_csv(dest_csv_file, sep=',', encoding='utf-8')


def print_data_frame(data_frame, print_columns=True, print_rows=True):
    # print column names
    if print_columns:
        print(",".join([column for column in data_frame]))

    if print_rows:
        for index, row in data_frame.iterrows():
            print(",".join([str(row[column]) for column in data_frame]))


if __name__ == '__main__':
    csv_file = 'Demographic_Statistics_By_Zip_Code.csv'
    print("Loading the csv file...")
    data_frame = read_csv(csv_file)
    print("loading the csv file successfully...")

    print_data_frame(data_frame, False, True)

    print("Write the dataframe as a csv file...")
    dest_csv_file = 'Demographic_Statistics_New.csv'
    write_in_csv(data_frame, dest_csv_file)
    print("Write csv file successfully...")
