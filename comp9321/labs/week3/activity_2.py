import pandas as pd

if __name__ == '__main__':
    csv_file = 'Books.csv'
    data_frame = pd.read_csv(csv_file)
    feature = "Place of Publication"
    # print(data_frame[feature])

    f = lambda x: 'London' if 'London' in x else x.replace('-', ' ')
    data_frame[feature] = data_frame[feature].apply(f)
    ################################################################################################################
    # Here is also another approach using numpy.where                                                              #
    #    import numpy as np                                                                                        #
    #    london = df['Place of Publication'].str.contains('London')                                                #
    #    df['Place of Publication'] = np.where(london, 'London', df['Place of Publication'].str.replace('-', ' ')) #
    ################################################################################################################
    # print(data_frame[feature])

    # We use Pandas' extract method which for each subject string in the Series,
    # extracts groups from the first match of regular expression pat.
    feature = "Date of Publication"
    new_date = data_frame[feature].str.extract(r'^(\d{4})', expand=False)
    new_date = pd.to_numeric(new_date, errors='ignore')
    new_date = new_date.fillna(0)
    data_frame[feature] = new_date
    print(data_frame[feature])
