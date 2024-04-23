# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:04:27 2024

@author: erwan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
import pvlib
from pvlib.location import Location
from pandas.plotting import register_matplotlib_converters


###############################################################################
###############################################################################
###############################################################################
###############################################################################
#______________Importing raw data______________________________________________
#
###############################################################################
###############################################################################
###############################################################################
###############################################################################

import xarray as xr
import netCDF4 as nc

from datetime import datetime, timedelta
import os.path, os
from siphon.catalog import TDSCatalog # websvr giving access to data and metadata
from urllib.error import HTTPError

import requests
from bs4 import BeautifulSoup

def liste_of_link(url):
    """
    Create a list which contains sub-lists corresponding
    to the different islands and in each sub-list there 
    is a list for each station on the island

    Args:
        url (str): lien of IOS-NET website.
        
    Returns:
        L (list) : list which contains sub-lists corresponding
        to the different islands and in each sub-list there 
        is a list for each station on the island
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
                    # Afficher les noms des fichiers .nc
                    print("Fichiers .nc disponibles :")
                    for nc_file in nc_files:
                        print(nc_file)
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




# Fonction de tri personnalisée ( permet de mettre les GHI en premier dans la liste )
def custom_sort(item):
    if 'GHI' in item:
        return (0, item)  # Mettre les chaînes contenant 'GHI' en premier
    else:
        return (1, item)  # Mettre les autres chaînes après



# Conversion dataframe
def nc_to_dataframe_second(ncfile):
    liste_variable = []
    for varname, var in ncfile.variables.items():
        if hasattr(var, "standard_name"):
            liste_variable.append(varname)

    # Filtrage des chaînes de caractères contenant 'DHI' ou 'GHI'
    resultat_0 = [var for var in liste_variable if 'DHI' in var or 'GHI' in var or 'DNI' in var]
    resultat = [var for var in resultat_0 if 'Avg' in var]
    # Réorganisation de la liste
    resultat_triee = sorted(resultat, key=custom_sort)

    # Extract the time and variables arrays from the netCDF file
    time_unix = ncfile.variables['time'][:]
    variable_arrays = [ncfile.variables[var][:] for var in resultat_triee]

    # Convert the time to pandas datetime format with UTC timezone and then to local time
    time_utc = pd.to_datetime(time_unix, unit='s', origin='unix', utc=True).tz_convert('Indian/Reunion')
    time_local = time_utc.tz_convert('Indian/Reunion')
    #print(time_local)

    # Create a dataframe with the variables arrays
    df = pd.DataFrame(dict(zip(resultat_triee, variable_arrays)), index=time_local)
    
    ncfile.close() # Close the NetCDF file properly
    
    return df

def nc_to_dataframe(nc_file_paths):
    """
    Make one dataframe with divers netCDF4 files

    Args:
        nc_file_paths (lst): list of netCDF4 files.
        
    Returns:
        final_df (Dataframe) : Compilation of all temporel data in one dataframe
    """
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

def affiche_data(liste):
    """
    Make one dataframe par station

    Args:
        liste (lst): list of netCDF4 files.
        
    Returns:
        name_station (str) : Name of station.
        df (Dataframe) : Compilation of all temporel data in one dataframe
    """
    name_station=liste[0]
    liste_data=liste[1:]
    df=nc_to_dataframe(liste_data)
    return name_station , df






###############################################################################
###############################################################################
###############################################################################
###############################################################################
#______________ Raw data processing ______________________________________________
#
###############################################################################
###############################################################################
###############################################################################
###############################################################################




#==============================================================================
# affichage des courbes ghi et dhi
#==============================================================================


#____faster method____

def affiche_courbe(name,df):
    """
    Plot the irradiance curves (DHI and GHI) for a given station.

    Args:
        name (str): Name of the station.
        df (DataFrame): DataFrame containing irradiance valuable data.
        
    Returns:
        None
    """
    # Convert the index to datetime
    df.index = pd.to_datetime(df.index)
    # Create the figure

    df.plot(figsize=(10, 8),linewidth=0.5)
    plt.xlabel('Time [UTC+4]')
    plt.ylabel('Irradiance [$W/m^2$]')
    plt.title(f'GHI and DHI in {name}')
    plt.legend(loc='best')
    
    
    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return 

def affiche_courbe_dhi_dni_ghi(name,df):
    
    """
    Plot the irradiance curves (DHI, DNI and GHI) for a given station.

    Args:
        name (str): Name of the station.
        df (DataFrame): DataFrame containing irradiance valuable data.
        
    Returns:
        None
    """
    # Convert the index to datetime
    df.index = pd.to_datetime(df.index)
    # Create the figure

    df['ghi'].plot(linewidth=0.5)
    df['dni'].plot(linewidth=0.5)
    df['dhi'].plot(linewidth=0.5)
    plt.xlabel('Time [UTC+4]')
    plt.ylabel('Irradiance [$W/m^2$]')
    plt.title(f'DHI, DNI and GHI in {name}')
    plt.legend(loc='best')
    
    
    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return 

#-----------------------------------------------------------------------------
#__interactive plot for jupyter notebook____
#-----------------------------------------------------------------------------

# for interactif graph use 'pip install plotly'

#pip install plotly


import plotly.graph_objects as go

def affiche_courbe_interact_dhi_dni_ghi(name, df):
    """
    Plot the irradiance curves (DHI, DNI and GHI) for a given station using Plotly.

    Args:
        name (str): Name of the station.
        df (DataFrame): DataFrame containing irradiance valuable data.
        
    Returns:
        None
    """
    # Convert the index to datetime
    #df.index = pd.to_datetime(df.index)
    
    # Create the figure
    fig = go.Figure(layout=dict(width=1000, height=600))

    
    
    # Add traces for each irradiance type
    fig.add_trace(go.Scatter(x=df.index, y=df['ghi'], mode='lines', name='GHI'))
    fig.add_trace(go.Scatter(x=df.index, y=df['dni'], mode='lines', name='DNI'))
    fig.add_trace(go.Scatter(x=df.index, y=df['dhi'], mode='lines', name='DHI'))
    
    # Customize layout
    fig.update_layout(title=f'DHI, DNI and GHI in {name}',xaxis_title='Time [UTC+4]',yaxis_title='Irradiance [W/m^2]')
    
    # Show the plot
    fig.show()
    return


'''
def courbe_ghi_dhi(name, df_station):
    """
    Plot the irradiance curves (DHI and GHI) for a given station.

    Args:
        name (str): Name of the station.
        df_station (DataFrame): DataFrame containing irradiance data.
        
    Returns:
        None
    """
    i=0
    # Convert the index to datetime
    df_station.index = pd.to_datetime(df_station.index)
    
    # Create the figure
    plt.figure(f'DHI and GHI for station {name}',figsize=(10, 5))
    #plt.clf() #remove the hashtag if you only want to display one window at a time
    
    
    # Plot the curves for each column
    for col in df_station.columns:
        i=i+1
        print(f'Colonne : {i}') 
        plt.plot(df_station.index, df_station[col], label=col, linewidth=0.5)
    
    # Add title and labels
    plt.title(f'DHI and GHI for station {name}')
    plt.xlabel('Time [UTC+4]')
    plt.ylabel('Irradiance [$W/m^2$]')
    plt.legend(loc='best')
    
    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

    
    
def affiche_courbe_bis(name,df):
    """
    Plot the irradiance curves (DHI and GHI) for a given station.

    Args:
        name (str): Name of the station.
        df (DataFrame): DataFrame containing irradiance valuable data.
        
    Returns:
        None
        
    Use this fonction after use the function "one_column_ghi_dhi(df_station)".
    """
    # Convert the index to datetime
    df.index = pd.to_datetime(df.index)
    # Create the figure
    plt.figure(f'DHI and GHI for station {name}',figsize=(10, 5))

    GHI=df['ghi']
    DHI=df['dhi']
    plt.plot(GHI, linestyle='-', linewidth=0.5, color='r', label='GHI')
    plt.plot(DHI, linestyle='-', linewidth=0.5, color='b', label='DHI')
    plt.xlabel('Time [UTC+4]')
    plt.ylabel('Irradiance [$W/m^2$]')
    plt.title(f'GHI and DHI in {name}')
    plt.legend(loc='best')
    
    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return 
'''    


#______________________________________________________________________________
# 
# Suivre les intructions dans l'ordre 
#______________________________________________________________________________
    
    
    

#==============================================================================
# show only one column of ghi and one column of dhi
#==============================================================================


def one_column_ghi_dhi(df_station):
    """
    Show only one column of ghi and one column of dhi.

    Args:
        df_station (DataFrame): DataFrame containing irradiance data.
        
    Returns:
        df (DataFrame): DataFrame containing irradiance data with only 2 columns.
    """
    df=df_station.copy()
    df.index = pd.to_datetime(df.index) # conversion of index
    # Filter columns containing 'GHI' and sum the values
    df['ghi'] = df.filter(like='GHI').sum(axis=1)
    
    # Filter columns containing 'DHI' and sum the values
    df['dhi'] = df.filter(like='DHI').sum(axis=1)

    # Filter columns containing 'DHI' and sum the values
    df['dni'] = df.filter(like='DNI').sum(axis=1)
    
    # Delete individual GHI and DHI columns
    df = df.drop(columns=df.filter(like='GHI').columns)
    df = df.drop(columns=df.filter(like='DHI').columns)
    df = df.drop(columns=df.filter(like='DNI').columns)
    df = df.replace(0,np.nan)
    return df



#==============================================================================
# Estimation of DNI with GHI and DHI
#==============================================================================


def estimation_dni(df, df_geo, name_station, time_zone):
    """
    Estimation of DNI values with GHI and DHI

    Args:
        df (DataFrame): DataFrame containing irradiance data (GHI and DHI).
        df_geo (DataFrame): DataFrame containing geographic data (Longitude, Latitude and Altitude).
        name_station (str): Name of the station.
        time_zone (str): time zone of station.
        
    Returns:
        df_1 (DataFrame): DataFrame containing irradiance data (dhi ,dni, ghi, mu0, extra_radiation and zenith).
    """
    df_1 = df.copy() # copy to use new dataframe
    #df_1.index = pd.to_datetime(df_1.index) # conversion of index
    good_data = df_geo.loc[df_geo.index[df_geo.index == f'{name_station}']] # select of geographic data of station
    a = good_data.iloc[0]
    # Definition of Location oject. Coordinates and elevation of La Plaine des Palmistes (Reunion)
    site = Location(a.Latitude, a.Longitude, time_zone, a.Altitude, f'{name_station} (SWOI)') # latitude, longitude, time_zone, altitude, name
    solpos = site.get_solarposition(df_1.index)
    df_1['zenith'] = solpos['zenith']
    df_1['extra_radiation'] = pvlib.irradiance.get_extra_radiation(df_1.index)
    df_1['mu0'] = np.cos(np.deg2rad(df_1['zenith'])).clip(lower=0.1)
    df_1['dni'] = (df_1['ghi'] - df_1['dhi'] ) / df_1['mu0']
    return df_1



#==============================================================================
# Deletes data above the physical limit
#==============================================================================


def quality_of_bsrn(df):
    """
    Delete data that does not meet BSRN criteria

    Args:
        df (DataFrame): DataFrame containing irradiance data.
        
    Returns:
        df_hourly_mean (DataFrame): DataFrame containing filter irradiance data with hourly mean.
    """
    # Create a copy of the DataFrame to avoid modifying the original data
    df_1 = df.copy()
    nb_nan_init = count_nan(df_1)

    # Replace outliers with np.nan for each measurement
    # For 'ghi'
    df_1.loc[(df_1['ghi'] < -4) | (df_1['ghi'] > 1.5 * df_1['extra_radiation'] * df_1['mu0']**1.2 + 100), 'ghi'] = np.nan
    # For 'dhi'
    df_1.loc[(df_1['dhi'] < -4) | (df_1['dhi'] > 0.95 * df_1['extra_radiation'] * df_1['mu0']**1.2 + 50), 'dhi'] = np.nan
    # For 'dni'
    df_1.loc[(df_1['dni'] < -4) | (df_1['dni'] > df_1['extra_radiation']), 'dni'] = np.nan

    nb_nan_final = count_nan(df_1)
    print(f'{nb_nan_final-nb_nan_init} delete values')

    # Group data by hour and calculate the mean
    df_hourly_mean = df_1.resample('H').mean()

    # Returns the DataFrame with outliers replaced by np.nan
    return df_hourly_mean

#==============================================================================
# Enumerate np.nan values in dataframe
#==============================================================================

def count_nan(df):
    """
    Compte le nombre de valeurs np.nan dans un DataFrame.

    Args :
        df (DataFrame) : DataFrame pandas

    Returns :
        count (Interger) : Number of np.nan values
    """
    # Compte le nombre de np.nan dans le DataFrame
    count = df.isnull().sum().sum()
    
    return count



###############################################################################
###############################################################################
###############################################################################
###############################################################################
#______________ Data comparison ______________________________________________
#
###############################################################################
###############################################################################
###############################################################################
###############################################################################

# librairie
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import sys


def show_irrad_sarah(data_geo,STANDARD_NAME, ncfile ,url , min_long,max_long,min_lat,max_lat):
    """
    Show the irradiance data on a map with netCDF file

    Args:
        data_geo (pandas.core.frame.DataFrame) : dataframe which contains all geographical data of stations
        STANDARD_NAME (str) : name of irradiance data
        ncfile (netCDF4._netCDF4.Dataset) : dataset in format netCFD4
        url (str) : netCDF file path
        min_long (int) : minimum value of longitude for the figure
        max_long (int) : maximum value of longitude for the figure
        min_lat (int) : minimum value of latitude for the figure
        max_lat (int) : maximum value of latitude for the figure
        
    Returns:
        None

    Example :
        data_geo = pd.read_csv('data_geo_all_station.csv', sep=';', index_col=0)
        STANDARD_NAME = "surface_direct_along_beam_shortwave_flux_in_air"
        ncfile = nc.Dataset(url_file_2020_01)
        url = "C:/.../data_sarah3/DNImm202001010000004UD1000101UD.nc"
        min_long = 20
        max_long = 65
        min_lat = -35
        max_lat = 0
    """
    # Searching for variable by standard name
    varname = next((name for name, var in ncfile.variables.items() if hasattr(var, 'standard_name') and var.standard_name == STANDARD_NAME), None)
    
    if not varname:
        print(f"Error: Unable to find variable with standard name '{STANDARD_NAME}' in file: {url}")
        ncfile.close()
        sys.exit(1)
    
    # Extracting parameters
    latitude = ncfile.variables['lat'][:]
    longitude = ncfile.variables['lon'][:]
    
    # Extracting time
    time_unix = ncfile.variables['time'][:]
    var_array = np.array(ncfile.variables[varname][:])
    var_array = var_array.astype(np.float32) # conversion  dtype=int16 to dtype=float32
    var_array[var_array == -999] = np.nan
    
    # Converting time to human-readable format
    time_array = np.array([datetime.fromtimestamp(t) for t in time_unix])

    mask_long = (longitude >= min_long) & (longitude <= max_long)
    mask_lat = (latitude >= min_lat) & (latitude <= max_lat)
    # Apply the mask to the longitude and latitude to solar flux data
    new_longitude = longitude[mask_long]
    new_latitude = latitude[mask_lat]

    # Apply the mask in irradiance data
    masked_var_array = var_array[:, mask_lat, :][:, :, mask_long]


    # Creat a map
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Draw data in map
    plt.contourf(new_longitude, new_latitude, masked_var_array[0], transform=ccrs.PlateCarree(),cmap='YlOrRd')
    
    # Add map details
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5, edgecolor='gray')
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.2, edgecolor='gray')
    ax.gridlines(draw_labels=True)

    # Add color bar
    plt.colorbar(label=f'{ncfile.variable_id} [W/$m^2$]')
    
    # Add a title
    plt.title(f'{ncfile.title}')

    
    # Add station meteo
    for i in range(data_geo.shape[0]):
        if min_long<data_geo.iloc[i,1]<max_long :
            if min_lat<data_geo.iloc[i,0]<max_lat:
                plt.scatter(data_geo.iloc[i,1], data_geo.iloc[i,0], color='blue', label=f'Station of {data_geo.index[i]}',s=30,marker='^')
    plt.legend(bbox_to_anchor=(1.25, 1.0), loc="upper left",fontsize=7)


    # Show map
    plt.show()
    print("Horizontale axis : Longitude \nVertical axis : Latitude")
    return

def show_irrad_sarah_pixel(maps,data_geo,STANDARD_NAME, ncfile ,url , min_long,max_long,min_lat,max_lat):
    """
    Show the irradiance data on a map with netCDF file

    Args:
        map (cartopy.crs.PlateCarree) : type of projection to map
        data_geo (pandas.core.frame.DataFrame) : dataframe which contains all geographical data of stations
        STANDARD_NAME (str) : name of irradiance data
        ncfile (netCDF4._netCDF4.Dataset) : dataset in format netCFD4
        url (str) : netCDF file path
        min_long (int) : minimum value of longitude for the figure
        max_long (int) : maximum value of longitude for the figure
        min_lat (int) : minimum value of latitude for the figure
        max_lat (int) : maximum value of latitude for the figure
        
    Returns:
        None

    Example :
        maps = ccrs.PlateCarree()
        data_geo = pd.read_csv('data_geo_all_station.csv', sep=';', index_col=0)
        STANDARD_NAME = "surface_direct_along_beam_shortwave_flux_in_air"
        url = "C:/.../data_sarah3/DNImm202001010000004UD1000101UD.nc"
        ncfile = nc.Dataset(url)
        min_long = 20
        max_long = 65
        min_lat = -35
        max_lat = 0
    """
    # Searching for variable by standard name
    varname = next((name for name, var in ncfile.variables.items() if hasattr(var, 'standard_name') and var.standard_name == STANDARD_NAME), None)
    
    if not varname:
        print(f"Error: Unable to find variable with standard name '{STANDARD_NAME}' in file: {url}")
        ncfile.close()
        sys.exit(1)
    
    # Extracting parameters
    latitude = ncfile.variables['lat'][:]
    longitude = ncfile.variables['lon'][:]
    
    # Extracting time
    time_unix = ncfile.variables['time'][:]
    var_array = np.array(ncfile.variables[varname][:])
    var_array = var_array.astype(np.float32) # conversion  dtype=int16 to dtype=float32
    var_array[var_array == -999] = np.nan
    
    # Converting time to human-readable format
    time_array = np.array([datetime.fromtimestamp(t) for t in time_unix])

    mask_long = (longitude >= min_long) & (longitude <= max_long)
    mask_lat = (latitude >= min_lat) & (latitude <= max_lat)
    # Apply the mask to the longitude and latitude to solar flux data
    new_longitude = longitude[mask_long]
    new_latitude = latitude[mask_lat]

    # Apply the mask in irradiance data
    masked_var_array = var_array[:, mask_lat, :][:, :, mask_long]

    # Creat a map
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=maps)
    
    # Tracer les données sur la carte avec pcolormesh
    pcm = ax.pcolormesh(new_longitude, new_latitude, masked_var_array[0],transform=maps,cmap='YlOrRd')
    
    # Add map details
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5, edgecolor='gray')
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.2, edgecolor='gray')
    ax.gridlines(draw_labels=True)
    
    # Ajouter une barre de couleur
    cbar = plt.colorbar(pcm, label=f'{ncfile.variable_id} [W/$m^2$]')

    # Add a title
    plt.title(f'{ncfile.title}')

    
    # Add station meteo
    for i in range(data_geo.shape[0]):
        if min_long<data_geo.iloc[i,1]<max_long :
            if min_lat<data_geo.iloc[i,0]<max_lat:
                plt.scatter(data_geo.iloc[i,1], data_geo.iloc[i,0], color='blue', label=f'Station of {data_geo.index[i]}',s=30,marker='^')
    plt.legend(bbox_to_anchor=(1.25, 1.0), loc="upper left",fontsize=7)


    # Show map
    plt.show()
    print("Horizontale axis : Longitude \nVertical axis : Latitude")
    return


# adaptation of geographical data


def data_geo_adaption(data_geo, url):
    """
    Adaption geographical data in-situe to SARAH-3 data

    Args:
        data_geo (dataframe) : dataframe which contains all geographical data of stations
        url (str) : netCDF file path
        
    Returns:
        data_geo_modif (dataframe) : dataframe which contains all geographical data of stations adapted

    Example :
        data_geo = pd.read_csv('data_geo_all_station.csv', sep=';', index_col=0)
        url = "C:/.../data_sarah3/DNImm202001010000004UD1000101UD.nc"
    """
    ncfile = nc.Dataset(url) #dataset in format netCFD4
    
    # Extracting parameters
    latitude = ncfile.variables['lat'][:]
    longitude = ncfile.variables['lon'][:]
    
    def approched_latitude(nombre):
        liste = latitude.data.tolist()
        l = len(liste)
        L = []
        for i in range(l):
            L.append(abs(liste[i]-nombre))
        a = L.index(min(L))
        return round(liste[a],4)

    def approched_longitude(nombre):
        liste = longitude.data.tolist()
        l = len(liste)
        L = []
        for i in range(l):
            L.append(abs(liste[i]-nombre))
        a = L.index(min(L))
        return round(liste[a],4)

    # Adaption of datafreame
    data_geo_modif = data_geo.copy()
    data_geo_modif['Latitude_modif'] = data_geo_modif['Latitude'].apply(approched_latitude)
    data_geo_modif['Longitude_modif'] = data_geo_modif['Longitude'].apply(approched_longitude)
    return data_geo_modif


