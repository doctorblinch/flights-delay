import streamlit as st
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json


with open('cities.json', 'r') as file:
        KNOWN_CITIES = json.load(file)

MAX_IMAGE_SIZE = 1080


def generate_photo(arrival_city, departure_city, size=(10,7)):
    plt.figure(figsize=size, frameon=False)
    map = Basemap(resolution='c', llcrnrlon=-180, urcrnrlon=-50,
              llcrnrlat=10, urcrnrlat=75, lat_0=0, lon_0=0)
    map.drawcoastlines()
    map.drawcountries(linewidth = 1)
    map.drawmapboundary(fill_color='aqua')
    map.drawstates(color='0.3')
    map.fillcontinents(color='green',lake_color='aqua')
    
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


    if st.button('Fly!'):
        generate_photo(arrival_city, departure_city)
        filename = 'images/map.png'
        try:
            with open(filename, 'rb') as input:
                st.image(input.read(), width=MAX_IMAGE_SIZE)
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
    airports = pd.read_csv('data/airports.csv')
    return data, airports


def statistics_page():
    st.title('Statistics 🎲')

    st.header('Count of flights recorded during year 2015 in each airport:')
    st.image('images/count_origin_airport.png', width=MAX_IMAGE_SIZE)
    
    data, airports = read_data()
    data['DATE'] = pd.to_datetime(data[['YEAR','MONTH', 'DAY']])


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

def random_forest_score():
    clf = RandomForestClassifier()
    clf = clf.fit(X, Y)
    pass


def get_X_y_from_data(data):
    kf = KFold(n_splits = 5)



if __name__ == '__main__':
    st.sidebar.title("Choose page:")
    pages = ['Presentation', 'Statistics', 'Code']
    page = st.sidebar.radio("Navigate", options=pages)

    if page == 'Presentation':
        presentation_page()

    if page == 'Statistics':
        statistics_page()
