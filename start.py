import streamlit as st
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians
import pickle
import json

from sklearn.preprocessing import LabelEncoder

with open('data/JSON/cities.json', 'r') as file:
    KNOWN_CITIES = json.load(file)

AIRLINES = ['American Airlines Inc.', 'Alaska Airlines Inc.', 'JetBlue Airways', 'Delta Air Lines Inc.',
            'Atlantic Southeast Airlines', 'Frontier Airlines Inc.', 'Hawaiian Airlines Inc.',
            'American Eagle Airlines Inc.', 'Spirit Air Lines', 'Skywest Airlines Inc.', 'United Air Lines Inc.',
            'US Airways Inc.', 'Virgin America', 'Southwest Airlines Co.']

MAX_IMAGE_SIZE = 1080


def generate_photo(arrival_city, departure_city, size=(10, 7)):
    plt.figure(figsize=size, frameon=False)
    map = Basemap(resolution='c', llcrnrlon=-180, urcrnrlon=-50,
                  llcrnrlat=10, urcrnrlat=75, lat_0=0, lon_0=0)
    map.drawcoastlines()
    map.drawcountries(linewidth=1)
    map.drawmapboundary(fill_color='aqua')
    map.drawstates(color='0.3')
    map.fillcontinents(color='green', lake_color='aqua')

    x, y = get_city_coordinate(arrival_city)
    plt.plot(x, y, 'ok', markersize=3)
    plt.text(x, y, arrival_city, fontsize=8, color='red')

    x, y = get_city_coordinate(departure_city)
    plt.plot(x, y, 'ok', markersize=3)
    plt.text(x, y, departure_city, fontsize=8, color='blue')

    plt.show()
    plt.savefig('images/map.png', dpi=200, bbox_inches='tight', pad_inches=0)


def get_city_coordinate(city: str):
    if city.lower() in KNOWN_CITIES:
        city = KNOWN_CITIES[city.lower()]
        return city['x'], city['y']

    return -70, 40


def presentation_page():
    st.title('Presentation layer')

    st.image('images/plane_departure.png', width=30)
    departure_city = st.selectbox('Departure city', list(KNOWN_CITIES.keys()))
    st.image('images/plane_arrival.png', width=30)
    arrival_city = st.selectbox('Arrival city', list(KNOWN_CITIES.keys()))

    airline = st.selectbox('Airline', AIRLINES)

    day_of_week = st.selectbox('Day', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    if st.button('Fly!'):
        generate_photo(arrival_city, departure_city)
        filename = 'images/map.png'
        try:
            with open(filename, 'rb') as input:
                st.image(input.read(), width=MAX_IMAGE_SIZE)
            prediction = machine_learning(arrival_city, departure_city, day_of_week, airline)
            st.write('Prediction: ')
            pd.set_option('display.max_columns', 15)
            st.dataframe(prediction.append(prediction))

        except FileNotFoundError:
            st.error('Default map file not found.')
    else:
        filename = 'images/base_map.png'
        try:
            with open(filename, 'rb') as input:
                st.image(input.read(), width=MAX_IMAGE_SIZE)
        except FileNotFoundError:
            st.error('File not found.')


@st.cache
def read_data():
    data = pd.read_csv('data/flights.csv')
    airports = pd.read_csv('data/airports_lower.csv')
    return data, airports


def machine_learning(arrival_city, departure_city, day_of_week, airline):
    with open('data/model', 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv('data/X.csv')
    sample = generate_sample(arrival_city, departure_city, day_of_week, airline)
    df = df.append(sample)

    le = LabelEncoder()
    df['AIRLINE'] = le.fit_transform(df['AIRLINE'])
    df['ORIGIN_AIRPORT'] = le.fit_transform(df['ORIGIN_AIRPORT'])
    df['DESTINATION_AIRPORT'] = le.fit_transform(df['DESTINATION_AIRPORT'])
    df['Day'] = le.fit_transform(df['Day'])

    y = np.array(df.iloc[-1])
    y = y.reshape(1, 12)
    prediction = model.predict(y)
    sample['PREDICTION'] = prediction
    return sample


def get_airport_of_city(city):
    airports = pd.read_csv('data/airports_lower.csv')
    return np.random.choice(airports[airports.CITY == city]['IATA_CODE'])


def get_distance_coordinates(lat1, lon1, lat2, lon2):
    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def get_distance_between_airports(port1, port2):
    airports = pd.read_csv('data/airports_lower.csv')
    lat1 = float(airports[airports.IATA_CODE == port1].LATITUDE)
    lon1 = float(airports[airports.IATA_CODE == port1].LONGITUDE)

    lat2 = float(airports[airports.IATA_CODE == port2].LATITUDE)
    lon2 = float(airports[airports.IATA_CODE == port1].LONGITUDE)
    return get_distance_coordinates(lat1, lon1, lat2, lon2)


def generate_sample(arrival_city, departure_city, day_of_week, airline):
    origin_airport = get_airport_of_city(departure_city)
    destination_airport = get_airport_of_city(arrival_city)
    distance = get_distance_between_airports(origin_airport, destination_airport)
    scheduled_time = distance / 7 + np.random.normal(0.75, 1.5)
    sample = pd.DataFrame({
        'AIRLINE': airline,
        'ORIGIN_AIRPORT': origin_airport,
        'DESTINATION_AIRPORT': destination_airport,
        'DISTANCE': distance,
        'Day': day_of_week,
        'DEPARTURE_DELAY': np.random.normal(0, 20),
        'ARRIVAL_DELAY': np.random.normal(0, 20),
        'SCHEDULED_TIME': scheduled_time,
        'ELAPSED_TIME': scheduled_time * (1 - np.random.uniform(0.02, 0.07)),
        'AIR_TIME': scheduled_time * (1 - np.random.uniform(0.15, 0.22)),
        'TAXI_IN': abs(np.random.normal(6, 2)),
        'TAXI_OUT': abs(np.random.normal(18, 4.5)),
    }, index=[0])
    return sample


def statistics_page():
    st.title('Statistics 🎲')

    st.header('Count of flights recorded during year 2015 in each airport:')
    st.image('images/count_origin_airport.png', width=MAX_IMAGE_SIZE)

    data, airports = read_data()
    data['DATE'] = pd.to_datetime(data[['YEAR', 'MONTH', 'DAY']])


def ml_comparation_page():
    st.title('Model comparation')
    st.markdown('''
- Lasso\n
Mean Absolute Error: 7.280310210167826\n
Mean Squared Error: 97.19065931031942\n
Root Mean Squared Error: 9.85853231015243\n
R2 :  0.9383657079258535\n

\n=============================================\n


- Linear Regression\n
Mean Absolute Error: 7.557731617798706e-07\n
Mean Squared Error: 8.73424175046178e-13\n
Root Mean Squared Error: 9.345716532434408e-07\n
R2 :  0.9999999999999994\n

\n=============================================\n

- Ridge\n
Mean Absolute Error: 0.0001748317346789472\n
Mean Squared Error: 5.4915919275906806e-08\n
Root Mean Squared Error: 0.00023434145872189754\n
R2 :  0.9999999999651746\n

\n=============================================\n

- Random forest Regressor\n
Mean Absolute Error: 0.9692158638475284\n
Mean Squared Error: 8.65024518225491\n
Root Mean Squared Error: 2.9411299159090047\n
R2 :  0.994514372658243\n

\n=============================================\n

- Decision Tree Regressor\n
Mean Absolute Error: 0.6346336061270655\n
Mean Squared Error: 4.2500657182965105\n
Root Mean Squared Error: 2.0615687517753343\n
R2 :  0.9973047842902328\n

\n=============================================\n

- Boosted Linear\n
Mean Absolute Error: 4.2904825784351026e-13\n
Mean Squared Error: 3.1562608489903546e-25\n
Root Mean Squared Error: 5.618060919027449e-13\n
R2 :  1.0\n

\n=============================================\n

- Boosted Lasso\n
Mean Absolute Error: 2.7801428023783727\n
Mean Squared Error: 14.761939788013647\n
Root Mean Squared Error: 3.842126987491908\n
R2 :  0.990638588986516\n

\n=============================================\n

- Boosted Ridge\n
Mean Absolute Error: 0.008304517164112676\n
Mean Squared Error: 0.00010965635964005485\n
Root Mean Squared Error: 0.010471693255632292\n
R2 :  0.9999999304604769\n

\n=============================================\n

- Bagged Linear\n
Mean Absolute Error: 8.317778800323808e-07\n
Mean Squared Error: 1.058633433112669e-12\n
Root Mean Squared Error: 1.0288991365107997e-06\n
R2 :  0.9999999999999993\n

\n=============================================\n

- Bagged Lasso\n
Mean Absolute Error: 7.281159722322449\n
Mean Squared Error: 97.22133941612657\n
Root Mean Squared Error: 9.8600882052914\n
R2 :  0.9383462518730229\n

\n=============================================\n

- Bagged Ridge\n
Mean Absolute Error: 0.0001749826316381057\n
Mean Squared Error: 5.5013007744727e-08\n
Root Mean Squared Error: 0.0002345485189565839\n
R2 :  0.999999999965113\n
''')


if __name__ == '__main__':
    st.sidebar.title("Choose page:")
    pages = ['Presentation', 'Statistics', 'ML comparation']
    page = st.sidebar.radio("Navigate", options=pages)

    if page == 'Presentation':
        presentation_page()

    if page == 'Statistics':
        statistics_page()

    if page == 'ML comparation':
        ml_comparation_page()
