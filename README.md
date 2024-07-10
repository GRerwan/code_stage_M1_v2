# Intership of Master Energy : Inter-comparison and validation against in-situ measurements of satellite estimates of incoming solar radiation for Indien Ocean

This projet is my code development from my internship in M1 Energie in University of Reunion Island.

**DATA** : 
+ In-situ measurements : Stations of IOS-net in SWIO (South Weast of Indien Ocean)
+ Satellite estimates : SARAH-3

**Temporal resolution** : 
+ 2008/12/01-2024/04/01
+ IOS-net : 1 min
+ SARAH-3 : 30 min

## RAW DATA


### Importation raw data
We can upload all in IOS-NET website [1] with this python code : 

```python
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

url = "https://galilee.univ-reunion.fr/thredds/catalog/dataStations/catalog.html"

def liste_of_link(url):
    """
    Import all netCDF files by IOS-NET website

    Args:
        url (str): url containing all links
        
    Returns:
        L (lst): list containing list of links per stations

    Example:
        url = "https://galilee.univ-reunion.fr/thredds/catalog/dataStations/catalog.html"
    """
    # Début du compteur de temps
    start_time = time.time()
    L=[]
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Trouver toutes les balises 'a' qui représentent les dossiers
        folder_links = soup.find_all('a', href=True)
        # Extraire les noms de dossier des liens
        Names_of_country_in_SWOI = [folder.text.strip().rstrip('/') for folder in folder_links if folder.text.strip().endswith('/')]
        # Afficher les noms de dossier
        print("Names of country in SWOI:",Names_of_country_in_SWOI)
    else:
        print("Erreur lors de la requête HTTP :", response.status_code)

    for i in range(len(Names_of_country_in_SWOI)):
        url_1 = f"https://galilee.univ-reunion.fr/thredds/catalog/dataStations/{Names_of_country_in_SWOI[i]}/catalog.html"
        link_data=f'https://galilee.univ-reunion.fr/thredds/dodsC/dataStations/{Names_of_country_in_SWOI[i]}'
        response_1 = requests.get(url_1)
        if response_1.status_code == 200:
            soup = BeautifulSoup(response_1.content, 'html.parser')
            # Trouver toutes les balises 'a' qui représentent les dossiers
            folder_links = soup.find_all('a', href=True)
            # Extraire les noms de dossier des liens
            folder_names = [folder.text.strip().rstrip('/') for folder in folder_links if folder.text.strip().endswith('/')]
            # Afficher les noms de dossier
            print("Noms des dossiers :",folder_names)
        else:
            print("Erreur lors de la requête HTTP :", response.status_code)
        sous_liste=[]
        for j in range(len(folder_names)):
            url_2 = f"https://galilee.univ-reunion.fr/thredds/catalog/dataStations/{Names_of_country_in_SWOI[i]}/{folder_names[j]}/catalog.html"
            link_data_1=f'{link_data}/{folder_names[j]}'
            response_2 = requests.get(url_2)
            if response_2.status_code == 200:
                soup = BeautifulSoup(response_2.content, 'html.parser')
                # Trouver toutes les balises 'a' qui représentent les dossiers
                folder_links = soup.find_all('a', href=True)
                # Extraire les noms de dossier des liens
                folder_years = [folder.text.strip().rstrip('/') for folder in folder_links if folder.text.strip().endswith('/')]
                # Afficher les noms de dossier
                print("Available years :",folder_years)
            else:
                print("Erreur lors de la requête HTTP :", response.status_code)
            sous_sous_liste=[]
            sous_sous_liste.append(folder_names[j])
            for k in range(len(folder_years)):
                url_3 = f"https://galilee.univ-reunion.fr/thredds/catalog/dataStations/{Names_of_country_in_SWOI[i]}/{folder_names[j]}/{folder_years[k]}/catalog.html"
                link_data_2=f'{link_data_1}/{folder_years[k]}'
                response_3 = requests.get(url_3)
                if response_3.status_code == 200:
                    soup_2 = BeautifulSoup(response_3.content, 'html.parser')
                    # Trouver tous les liens se terminant par ".nc"
                    nc_links = soup_2.find_all('a', href=lambda href: href and href.endswith('.nc'))
                    # Extraire les noms des fichiers .nc
                    nc_files = [nc_link.text.strip() for nc_link in nc_links]
                    '''
                    # Afficher les noms des fichiers .nc
                    print("Fichiers .nc disponibles :")
                    for nc_file in nc_files:
                        print(nc_file)
                    '''
                else:
                    print("Erreur lors de la requête HTTP :", response_2.status_code)
                
                for l in range(len(nc_files)):
                    link_data_3=f'{link_data_2}/{nc_files[l]}'
                    sous_sous_liste.append(f'{link_data_3}')
            sous_liste.append(sous_sous_liste)
        L.append(sous_liste)
    # Fin du compteur de temps
    end_time = time.time()
    # Calcul du temps écoulé
    execution_time = end_time - start_time
    print("Temps d'exécution:", execution_time, "secondes")
    return L
liste_of_data=liste_of_link(url)


# data of South Africa
south_africa = liste_of_data[0]
#print(f'Data of South Africa : {south_africa}')

# data of Seychelles
seychelle = liste_of_data[1]
#print("\n")
#print(f'Data of Seychelles : {seychelle}')


# data of Mauritius
mauritius = liste_of_data[2]
#print("\n")
#print(f'Data of Mauritius : {mauritius}')


# data of Madagascar
mada = liste_of_data[3]
#print("\n")
#print(f'Data of Madagascar : {mada}')

# data of La Réunion
reunion = liste_of_data[4]
#print("\n")
#print(f'Data of Reunion : {reunion}')

# data of Comores
comores = liste_of_data[5]
#print("\n")
#print(f'Data of Comores : {comores}')


list_of_dat_per_country = [south_africa,seychelle,mauritius,mada,reunion,comores]
name_country = ['south_africa','seychelle','mauritius','mada','reunion','comores']
#############################################################################################################
def calculer_temps_execution(function, *args, **kwargs):
    # Record the start time
    start = time.time()
    
    # Call the function with provided arguments and keyword arguments
    result = function(*args, **kwargs)
    
    # Record the end time
    end = time.time()
    
    # Calculate the execution time
    execution_time = end - start
    
    # Print the execution time
    print(f"Execution time of {function.__name__}: {execution_time} seconds")
    
    return result

def save_all_raw_data_to_csv_file(list_of_dat_per_country,name_country):
    L = len(name_country)
    for i in range(L):
        k = len(list_of_dat_per_country[i])
        for j in range(k):
            name_station , df_station = affiche_data_reunion(list_of_dat_per_country[i][j])
            print(name_station)
            df_station.to_csv(f'data_raw_UTC/{name_country[i]}/{name_station}_irrad.csv', sep=';', index=True) 
    return


calculer_temps_execution(save_all_raw_data_to_csv_file,list_of_dat_per_country,name_country)      


### Collect GHI and DHI

### Estimate DNI with GHI and DHI








## Importation of satellite estimates data


# Source
[1] : https://galilee.univ-reunion.fr/thredds/catalog.html
