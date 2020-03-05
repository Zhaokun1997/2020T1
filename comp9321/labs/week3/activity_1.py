import pandas as pd


def print_data_frame(data_frame, print_columns=True, print_rows=True):
    if print_columns:
        print(",".join([column for column in data_frame]))
    if print_rows:
        for index, row in data_frame.iterrows():
            print(",".join([str(row[column]) for column in data_frame]))


if __name__ == '__main__':
    drop_list = ['Edition Statement',
                 'Corporate Author',
                 'Corporate Contributors',
                 'Former owner',
                 'Engraver',
                 'Contributors',
                 'Issuance type',
                 'Shelfmarks']
    csv_file = 'Books.csv'
    data_frame = pd.read_csv(csv_file)
    columns = [column for column in data_frame]

    # Calculate and print the number of nan (not a number) in each column
    print("******************************************************")
    num_of_rows = data_frame.shape[0]
    for c in columns:
        percent = 100 * data_frame[c].isna().sum() / num_of_rows
        print(c, str(percent) + '%')
    print("******************************************************")
    print("******************************************************")
    print_data_frame(data_frame, print_columns=True, print_rows=True)
    print("current shape is : ", data_frame.shape)
    print("******************************************************")

    # Drop the columns of dataframe in the drop_list
    # Pandas' drop method is used to remove columns of a dataframe
    # Inplace=True indicates that the changes should be applied to the given dataframe instead of creating a new one
    # axis=1 : Whether to drop labels from the index (0 / 'index') or columns (1 / 'columns').

    # data_frame.drop(columns=drop_list)  # change is not applied to the given data_frame
    data_frame.drop(drop_list, inplace=True, axis=1)
    print("******************************************************")
    print_data_frame(data_frame)
    print("after drop the shape is : ", data_frame.shape)
    print("******************************************************")
