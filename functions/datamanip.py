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
    # Filtrer les colonnes contenant 'GHI' et sommer les valeurs
    df['ghi'] = df.filter(like='GHI').sum(axis=1)
    
    # Filtrer les colonnes contenant 'DHI' et sommer les valeurs
    df['dhi'] = df.filter(like='DHI').sum(axis=1)
    
    # Suppression des colonnes individuelles GHI et DHI
    df = df.drop(columns=df.filter(like='GHI').columns)
    df = df.drop(columns=df.filter(like='DHI').columns)
    
    df = df.replace(0, np.nan)
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
    df_1.index = pd.to_datetime(df_1.index) # conversion of index
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

def estimation_dni_bis(df, df_geo, name_station, time_zone):
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
    df_1.index = pd.to_datetime(df_1.index) # conversion of index
    good_data = df_geo.loc[df_geo.index[df_geo.index == f'{name_station}']] # select of geographic data of station
    a = good_data.iloc[0]
    # Definition of Location oject. Coordinates and elevation of La Plaine des Palmistes (Reunion)
    site = Location(a.Latitude, a.Longitude, time_zone, a.Altitude, f'{name_station} (SWOI)') # latitude, longitude, time_zone, altitude, name
    solpos = site.get_solarposition(df_1.index)
    df_1['zenith'] = solpos['zenith']
    df_1['extra_radiation'] = pvlib.irradiance.get_extra_radiation(df_1.index)
    df_1['mu0'] = np.cos(np.deg2rad(df_1['zenith'])).clip(lower=0.1)
    df_1['dni'] = (df_1['ghi'] - df_1['dhi'] ) / np.cos(df_1['zenith'])
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
    # Crée une copie du DataFrame pour éviter de modifier les données d'origine
    df_1 = df.copy()

    # Remplacement des valeurs aberrantes par np.nan pour chaque mesure
    # Pour 'ghi'
    df_1.loc[(df_1['ghi'] < -4) | (df_1['ghi'] > 1.5 * df_1['extra_radiation'] * df_1['mu0']**1.2 + 100), 'ghi'] = np.nan
    # Pour 'dhi'
    df_1.loc[(df_1['dhi'] < -4) | (df_1['dhi'] > 0.95 * df_1['extra_radiation'] * df_1['mu0']**1.2 + 50), 'dhi'] = np.nan
    # Pour 'dni'
    df_1.loc[(df_1['dni'] < -4) | (df_1['dni'] > df_1['extra_radiation']), 'dni'] = np.nan

    # Regrouper les données par heure et calculer la moyenne
    df_hourly_mean = df_1.resample('H').mean()

    # Renvoie le DataFrame avec les valeurs aberrantes remplacées par np.nan
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
