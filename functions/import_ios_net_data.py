##################################### Librairy utile ##########################################################
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
#############################################################################################################

#############################################################################################################
def count_nan(df):
    """
    Count the number of np.nan values in a DataFrame.

    Args :
        df (DataFrame) : pandas DataFrame

    Returns :
        count (Integer) : Number of np.nan values
    """
    # Count the number of np.nan in the DataFrame
    count = df.isnull().sum().sum()
    
    return count

#############################################################################################################
# Fonction de tri personnalisée ( permet de mettre les GHI en premier dans la liste )
def custom_sort(item):
    if 'GHI' in item:
        return (0, item)  # Mettre les chaînes contenant 'GHI' en premier
    else:
        return (1, item)  # Mettre les autres chaînes après


#############################################################################################################
# Conversion dataframe
def nc_to_dataframe_second(ncfile):
    liste_variable = []
    for varname, var in ncfile.variables.items():
        if hasattr(var, "standard_name"):
            liste_variable.append(varname)

    # Filtrage des chaînes de caractères contenant 'DHI' ou 'GHI'
    resultat_1 = [var for var in liste_variable if 'DHI' in var or 'GHI' in var or 'DNI' in var]
    resultat = [var for var in resultat_1 if 'Avg' in var]
    
    # Réorganisation de la liste
    resultat_triee = sorted(resultat, key=custom_sort)

    # Extract the time and variables arrays from the netCDF file
    time_unix = ncfile.variables['time'][:]
    variable_arrays = [ncfile.variables[var][:] for var in resultat_triee]

    # Convert the time to pandas datetime format with UTC timezone and then to local time
    time_utc = pd.to_datetime(time_unix, unit='s', origin='unix', utc=False)

    # Create a dataframe with the variables arrays
    df = pd.DataFrame(dict(zip(resultat_triee, variable_arrays)), index=time_utc)
    
    ncfile.close() # Close the NetCDF file properly
    
    return df
#############################################################################################################
def nc_to_dataframe(nc_file_paths):
    # Initialiser une liste pour stocker les DataFrames pour chaque fichier .nc
    dfs = []
    # Parcourir chaque lien .nc dans la liste
    for nc_file_path in reversed(nc_file_paths):
        ncfile = nc.Dataset(nc_file_path)
        df = nc_to_dataframe_second(ncfile)

        # Convertir les données en un DataFrame et l'ajouter à la liste
        dfs.append(df)

    # Concaténer tous les DataFrames en un seul DataFrame
    final_df = pd.concat(dfs)

    return final_df
#############################################################################################################
def affiche_data_reunion(liste):
    name_station=liste[0]
    liste_data=liste[1:]
    df=nc_to_dataframe(liste_data)
    return name_station , df
#############################################################################################################
url = "https://galilee.univ-reunion.fr/thredds/catalog/dataStations/catalog.html"
#############################################################################################################
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

#############################################################################################################
def save_all_raw_data_to_csv_file(list_of_dat_per_country):
    L = len(list_of_dat_per_country)
    for i in range(L):
        k = len(list_of_dat_per_country[i])
        for j in range(k):
            name_station , df_station = affiche_data_reunion(list_of_dat_per_country[i][j])
            df_station.to_csv(f'data_raw_UTC_minute/{name_station}_irrad.csv', sep=';', index=True) 
            print(name_station)
    return
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