import pandas as pd


def print_data_frame(data_frame, print_columns=True, print_rows=True):
    if print_columns:
        print(",".join(data_frame.columns))
    if print_rows:
        for index, row in data_frame.iterrows():
            print(",".join([str(row[column]) for column in data_frame]))


def data_cleaning(data_frame):
    data_frame['Place of Publication'] = data_frame['Place of Publication'].apply(
        lambda x: 'London' if 'London' in x else x.replace('-', ' '))
    new_date = data_frame['Date of Publication'].str.extract(r'^(\d{4})', expand=False)
    new_date = pd.to_numeric(new_date)
    new_date = new_date.fillna(0)
    data_frame['Date of Publication'] = new_date
    return data_frame


if __name__ == '__main__':
    csv_file = 'Books.csv'
    df = pd.read_csv(csv_file)
    df = data_cleaning(df)

    # Replace the spaces with the underline character ('_')
    # Because panda's query method does not work well with column names which contains white spaces
    df.columns = [c.replace(' ', '_') for c in df.columns]

    # query
    df = df.query('Date_of_Publication > 1866 and Place_of_Publication == "London"')
    print_data_frame(df, True, True)
