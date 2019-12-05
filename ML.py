import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split


def read_data():
    data = pd.read_csv('data/flights.csv')
    airports = pd.read_csv('data/airports.csv')
    return data, airports


def read_processed_data(file_name='data/final_data.csv'):
    data = pd.read_csv(file_name)
    return data


def comparation_of_ML_models():
    data = read_processed_data()
    X = data.drop('ARRIVAL_DELAY', axis=1)
    y = data['ARRIVAL_DELAY']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Linear
    #lm = linear_model.LinearRegression()
    import sklearn.ensemble
    rf = sklearn.ensemble.RandomForestRegressor()
    model = rf.fit(X_train, y_train)
    print('Score:', model.score(X_test, y_test))

    return 10


def preprocess_data():
    # pd.set_option('display.max_columns', 15)
    data, airports = read_data()
    processed_data = data.copy()
    processed_data['DATE'] = pd.to_datetime(processed_data[['YEAR', 'MONTH', 'DAY']])
    processed_data.drop(['YEAR', 'MONTH', 'DAY'], axis=1, inplace=True)
    processed_data.drop(['FLIGHT_NUMBER'], axis=1, inplace=True)
    processed_data['SCHEDULED_DEPARTURE'] = create_flight_time(processed_data, 'SCHEDULED_DEPARTURE')
    processed_data['SCHEDULED_ARRIVAL'] = processed_data['SCHEDULED_ARRIVAL'].apply(format_heure)
    # processed_data['DEPARTURE_TIME'] = processed_data['DEPARTURE_TIME'].apply(format_heure)
    # processed_data['ARRIVAL_TIME'] = processed_data['ARRIVAL_TIME'].apply(format_heure)

    processed_data.drop(['DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', ], axis=1, inplace=True)
    processed_data.drop(['WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME',
                         'AIR_TIME', 'WHEELS_ON', 'TAXI_IN', # , 'ARRIVAL_TIME', 'ARRIVAL_DELAY'
                         'CANCELLATION_REASON', 'DIVERTED', 'CANCELLED',
                         'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
                         'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'], axis=1, inplace=True)

    # Split scheduled_time (time of the flight and distance into categorical variables
    # (example: short trip, shorter trip, medium, longer trip, long trip)
    # pd.cut(data['SCHEDULED_TIME'], 5, labels=[0,1,2,3,4])
    # pd.cut(data['DISTANCE'], 5, labels=[0,1,2,3,4])

    data['TRIP_LENGTH'] = (
        np.array(pd.qcut(data['DISTANCE'], 5, [0, 1, 2, 3, 4], retbins=True)[0]) + (
            np.array(pd.qcut(data['SCHEDULED_TIME'], 5, [0, 1, 2, 3, 4], retbins=True)[0]))
    )

    processed_data = data[np.isfinite(data['ARRIVAL_DELAY'])]
    processed_data.to_csv('data/final_data.csv', index=False)


# Function that convert the 'HHMM' string to datetime.time
def format_heure(chaine):
    if pd.isnull(chaine):
        return np.nan
    else:
        if chaine == 2400: chaine = 0
        chaine = "{0:04d}".format(int(chaine))
        heure = datetime.time(int(chaine[0:2]), int(chaine[2:4]))
        return heure


# _____________________________________________________________________
# Function that combines a date and time to produce a datetime.datetime
def combine_date_heure(x):
    if pd.isnull(x[0]) or pd.isnull(x[1]):
        return np.nan
    else:
        return datetime.datetime.combine(x[0], x[1])


# _______________________________________________________________________________
# Function that combine two columns of the dataframe to create a datetime format
def create_flight_time(df, col):
    liste = []
    for index, cols in df[['DATE', col]].iterrows():
        if pd.isnull(cols[1]):
            liste.append(np.nan)
        elif float(cols[1]) == 2400:
            cols[0] += datetime.timedelta(days=1)
            cols[1] = datetime.time(0, 0)
            liste.append(combine_date_heure(cols))
        else:
            cols[1] = format_heure(cols[1])
            liste.append(combine_date_heure(cols))
    return pd.Series(liste)
