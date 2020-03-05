import pandas as pd


def print_data_frame(data_frame, print_columns=True, print_rows=True):
    if print_columns:
        print(",".join(df.columns))
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
    books_csv_file = 'Books.csv'
    books_df = pd.read_csv(books_csv_file)
    books_df = data_cleaning(books_df)
    # Replace the spaces with the underline character ('_')
    # Because panda's query method does not work well with column names which contains white spaces
    books_df.columns = [c.replace(' ', '_') for c in books_df.columns]

    city_csv_file = 'City.csv'
    city_df = pd.read_csv(city_csv_file)

    # merge two dataframes
    df = pd.merge(books_df, city_df, how='left', left_on=['Place_of_Publication'], right_on=['City'])

    # Group by Country and keep the country as a column
    gb_df = df.groupby('Country', as_index=False)

    # Select a column (as far as it has values for all rows, you can select any column)
    result = gb_df['Identifier'].count()
    print_data_frame(result)
