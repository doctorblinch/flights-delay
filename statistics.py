from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict


def count_origin_airport():
    data = pd.read_csv('data/flights.csv')
    airports = pd.read_csv('data/airports.csv')
    count_flights = data['ORIGIN_AIRPORT'].value_counts()
    #___________________________
    plt.figure(figsize=(8,8))
    #________________________________________
    # define properties of markers and labels
    colors = ['yellow', 'red', 'lightblue', 'purple', 'green', 'orange']
    size_limits = [1, 100, 1000, 10000, 100000, 1000000]
    labels = []
    for i in range(len(size_limits)-1):
        labels.append("{} <.< {}".format(size_limits[i], size_limits[i+1])) 
    #____________________________________________________________
    map = Basemap(resolution='i',llcrnrlon=-180, urcrnrlon=-50,
                llcrnrlat=10, urcrnrlat=75, lat_0=0, lon_0=0,)
    map.shadedrelief()
    map.drawcoastlines()
    map.drawcountries(linewidth = 3)
    map.drawstates(color='0.3')
    #_____________________
    # put airports on map
    for index, (code, y,x) in airports[['IATA_CODE', 'LATITUDE', 'LONGITUDE']].iterrows():
        x, y = map(x, y)
        isize = [i for i, val in enumerate(size_limits) if val < count_flights[code]]
        ind = isize[-1]
        map.plot(x, y, marker='o', markersize = ind+5, markeredgewidth = 1, color = colors[ind],
                markeredgecolor='k', label = labels[ind])
    #_____________________________________________
    # remove duplicate labels and set their order
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    key_order = ('1 <.< 100', '100 <.< 1000', '1000 <.< 10000',
                '10000 <.< 100000', '100000 <.< 1000000')
    new_label = OrderedDict()
    for key in key_order:
        new_label[key] = by_label[key]
    plt.legend(new_label.values(), new_label.keys(), loc = 1, prop= {'size':11},
            title='Number of flights per year', frameon = True, framealpha = 1)
    plt.savefig('images/count_origin_airport.png', dpi=150, bbox_inches='tight', pad_inches=0)
