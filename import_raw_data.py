# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:04:31 2024

@author: erwan
"""




import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import sys
from datetime import datetime
from siphon.catalog import TDSCatalog

import netCDF4 as nc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr # aide la convertion en dataframe 

from datetime import datetime, timedelta
import os.path, os
from siphon.catalog import TDSCatalog # websvr giving access to data and metadata
from urllib.error import HTTPError

import requests
from bs4 import BeautifulSoup

import time

from functions import datamanip



# link of all SWOI data

url = "https://galilee.univ-reunion.fr/thredds/catalog/dataStations/catalog.html"



liste_of_data=datamanip.liste_of_link(url)


# data of South Africa
south_africa = liste_of_data[0]
print(f'Data of South Africa : {south_africa}')

# data of Seychelles
seychelle = liste_of_data[1]
print("\n")
print(f'Data of Seychelles : {seychelle}')


# data of Mauritius
mauritius = liste_of_data[2]
print("\n")
print(f'Data of Mauritius : {mauritius}')


# data of Madagascar
mada = liste_of_data[3]
print("\n")
print(f'Data of Madagascar : {mada}')

# data of Comores
comores = liste_of_data[5]
print("\n")
print(f'Data of Comores : {comores}')


# data of La Réunion
reunion = liste_of_data[4]
print("\n")
print(f'Data of Reunion : {reunion}')


# show curve of data

name_station_12, df_station_12 = datamanip.affiche_data(reunion[3])
print(f'Station of {name_station_12}')


