# library
import glob
import os
import seaborn as sns
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pvlib
from pvlib.location import Location
from pandas.plotting import register_matplotlib_converters
import plotly.graph_objects as go
import xarray as xr
import netCDF4 as nc
import sys
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

#############################################################################################################
# Désactiver l'affichage des FutureWarnings temporairement
warnings.simplefilter(action='ignore', category=FutureWarning)
#############################################################################################################


#############################################################################################################
#sns.set(style="darkgrid", palette="bright",context = "paper",font="Arial" )
sns.set(style="darkgrid", palette="bright",context = "paper",font="Arial" )
#############################################################################################################

#############################################################################################################
def hourly_mean(df):
    data = df.copy()
    data.index = pd.to_datetime(data.index)
    # Filtrer les heures UTC entre 02h et 14h (inclus)
    df_filtered = data.between_time('02:00', '14:00')
    # Group data by hour and calculate the mean
    df_hourly_mean = df_filtered.resample('H').mean()
    return df_hourly_mean
#############################################################################################################
def daily_mean(df):
    data = df.copy()
    data.index = pd.to_datetime(data.index)
    # Filtrer les heures entre 01h et 15h (inclus)
    df_filtered = data.between_time('02:00', '14:00')
    # Group data by hour and calculate the mean
    df_daily_mean = df_filtered.resample('D').mean()
    return df_daily_mean
#############################################################################################################
def monthly_mean(df):
    data = df.copy()
    data.index = pd.to_datetime(data.index)
    df_mounthly_mean = data.resample('M').mean()
    return df_mounthly_mean
#############################################################################################################
def yearly_mean(df):
    data = df.copy()
    data.index = pd.to_datetime(data.index)
    df_yearly_mean = data.resample('Y').mean()
    return df_yearly_mean
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
    time_utc = pd.to_datetime(time_unix, unit='s', origin='unix', utc=True).tz_convert('Indian/Reunion')
    time_local = time_utc.tz_convert('Indian/Reunion')
    #print(time_local)

    # Create a dataframe with the variables arrays
    df = pd.DataFrame(dict(zip(resultat_triee, variable_arrays)), index=time_local)
    
    ncfile.close() # Close the NetCDF file properly
    
    return df


#############################################################################################################
def nc_to_dataframe(nc_file_paths):
    # Initialiser une liste pour stocker les DataFrames pour chaque fichier .nc
    dfs = []
    i=0
    # Parcourir chaque lien .nc dans la liste
    for nc_file_path in reversed(nc_file_paths):
        i=i+1
        # Charger les données NetCDF en utilisant xarray
        print(f"tour: {i}/{len(nc_file_paths)}")
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
def courbe_ghi_dhi(liste):
    name , df_station = affiche_data_reunion(liste)
    i=0
    ligne , colon = df_station.shape
    plt.figure(figsize=(10, 5))  # Taille de la figure
    for col in df_station.columns:
        i=i+1
        print(f'Colonne :{i}/{colon}') 
        plt.plot(df_station.index, df_station[col], label=col)
        plt.legend(loc='best')
    
    # Ajouter le titre et les légendes
    plt.title(f'DHI and GHI for sation of {name}')
    plt.xlabel('Time [UTC+4]')
    plt.ylabel('Irradiance [$W/m^2$]')
    
    # Afficher le graphique
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return 
#############################################################################################################






#############################################################################################################
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

    df.plot(figsize=(10, 6),linewidth=0.5)
    plt.xlabel('Time [UTC+4]')
    plt.ylabel('Irradiance [$W/m^2$]')
    plt.title(f'Irridiance in {name} station')
    plt.legend(loc='best')
    
    
    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return 

#############################################################################################################
def one_column_ghi_dhi(df_station):
    """
    Show only one column of ghi and one column of dhi.

    Args:
        df_station (DataFrame): DataFrame containing irradiance data.
        
    Returns:
        df (DataFrame): DataFrame containing irradiance data with only 'ghi', 'dhi' and 'dni_ground' columns.
    """
    df = df_station.copy()

    # Convert index to datetime
    df.index = pd.to_datetime(df.index)
    
    # Filtrer les colonnes contenant 'GHI'
    ghi_columns = df.filter(like='GHI').columns
    dhi_columns = df.filter(like='DHI').columns
    dni_columns = df.filter(like='DNI').columns
    
    # Initialiser la colonne 'ghi' avec les valeurs de la première colonne trouvée
    df['ghi'] = df[ghi_columns[0]]
    df['dhi'] = df[dhi_columns[0]]
    
    if len(dni_columns)!=0:
        df['dni_ground'] = df[dni_columns[0]]
    
    if len(ghi_columns) > 1 : 
        for col in ghi_columns[1:]:
            df['ghi'] = df['ghi'].combine_first(df[col])
    else : 
        df = df
    
    if len(dhi_columns) > 1 : 
        for col in dhi_columns[1:]:
            df['dhi'] = df['dhi'].combine_first(df[col])
    else : 
        df = df
    
    if len(dni_columns) > 1 : 
        for col in dni_columns[1:]:
            df['dni_ground'] = df['dni_ground'].combine_first(df[col])
    else : 
        df = df
    
    # Remove the old columns
    df = df.drop(columns=df.filter(like='GHI').columns)
    df = df.drop(columns=df.filter(like='DHI').columns)
    df = df.drop(columns=df.filter(like='DNI').columns)
    
    
    return df

#############################################################################################################
def estimation_dni_physical_limits(df, df_geo, name_station, time_zone):
    """
    Estimation of DNI values with GHI and DHI

    Args:
        df (DataFrame): DataFrame containing irradiance data (GHI and DHI).
        df_geo (DataFrame): DataFrame containing geographic data (Longitude, Latitude and Altitude).
        name_station (str): Name of the station.
        time_zone (str): time zone of station.
        
    Returns:
        df_1 (DataFrame): DataFrame containing irradiance data (dhi ,dni, ghi, mu0, extra_radiation and zenith).

    Example:
        estimation_dni_physical_limits(df_test, data_geo, 'plaineparcnational', 'Indian/Reunion')
    """
    df_1 = df.copy() # copy to use new dataframe
    #df_1.index = pd.to_datetime(df_1.index) # conversion of index
    good_data = df_geo.loc[df_geo.index[df_geo.index == f'{name_station}']] # select of geographic data of station
    a = good_data.iloc[0]
    # Definition of Location object. Coordinates and elevation of La Plaine des Palmistes (Reunion)
    site = Location(a.Latitude, a.Longitude, time_zone, a.Altitude, f'{name_station} (SWOI)') # latitude, longitude, time_zone, altitude, name
    solpos = site.get_solarposition(df_1.index)
    df_1['SZA'] = solpos['zenith']
    df_1['extra_radiation'] = pvlib.irradiance.get_extra_radiation(df_1.index)
    df_1['mu0'] = np.cos(np.deg2rad(df_1['SZA'])).clip(lower=0.1)
    df_1['dni'] = (df_1['ghi'] - df_1['dhi'] ) / df_1['mu0']
    df_1['physical_limit_ghi'] = 1.5 * df_1['extra_radiation'] * df_1['mu0']**1.2 + 100
    df_1['physical_limit_dhi'] = 0.95 * df_1['extra_radiation'] * df_1['mu0']**1.2 + 50
    df_1['physical_limit_dni'] = df_1['extra_radiation']
    return df_1
    
#############################################################################################################
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
    df_1.index = pd.to_datetime(df_1.index) # conversion of index
    # Replace outliers with np.nan for each measurement
    # For 'ghi'
    df_1.loc[(df_1['ghi'] < -4) | (df_1['ghi'] > df_1['physical_limit_ghi']), 'ghi'] = np.nan
    # For 'dhi'
    df_1.loc[(df_1['dhi'] < -4) | (df_1['dhi'] > df_1['physical_limit_dhi']), 'dhi'] = np.nan
    # For 'dni'
    df_1.loc[(df_1['dni'] < -4) | (df_1['dni'] > df_1['extra_radiation']), 'dni'] = np.nan
    
    if 'dni_ground' in df_1.columns :
        df_1.loc[(df_1['dni_ground'] < -4) | (df_1['dni_ground'] > df_1['extra_radiation']), 'dni_ground'] = np.nan

    # Group data by hour and calculate the mean
    df_hourly_mean = hourly_mean(df_1)

    # Returns the DataFrame with outliers replaced by np.nan
    return df_hourly_mean

#############################################################################################################
def dni_data(df, data_geo,name,time_zone):
    '''
    Return a dataframe with DNI data

    Args :
        df (dataFrame) : pandas DataFrame without dni
        data_geo (dataFrame) : pandas Dataframe with gegraphical data
        name (str) : name of station on IOS-NET
        time_zone (str) : name of time zone

        
    Returns :
        filter_data (dataFrame) : pandas DataFrame contains irradiance filter data

    Example : 
        df = pd.read_csv('data_brute/reunion/plaineparcnational_irrad.csv', sep=';', index_col=0)
        data_geo = pd.read_csv('data_geo_all_station.csv', sep=';', index_col=0)
        name = 'plaineparcnational'
        time_zone = 'Indian/Reunion'
    '''
    # Record the start time
    start = time.time()
    
    df_one_columns = one_column_ghi_dhi(df)
    data_dni = estimation_dni_physical_limits(df_one_columns, data_geo, name, 'Indian/Reunion')
    filter_data = quality_of_bsrn(data_dni)
    # Delete unnecessary column
    #filter_data.drop(columns=['SZA',	'extra_radiation',	'mu0'], inplace=True)
    
    # Record the end time
    end = time.time()
    
    # Calculate the execution time
    execution_time = end - start
    
    # Print the execution time
    print(f"Execution: {execution_time} seconds")
    return filter_data


#############################################################################################################
def limits_BSRN_GHI(df_data):
    """
    Show mean value of irradiance data for each SWOI stations with Scatter plot

    Args:
        df_data (DataFrame): DataFrame containing mean irradiance data.
        
    Returns:
        None
    """
    df = df_data.copy()

    # Create the figure
    fig = go.Figure(layout=dict(width=1000, height=600))


    fig.add_trace(go.Scatter(x=df['SZA'], y=df['ghi'], mode='markers', name='Values',line=dict(width=1)))
    
    
    # Customize layout
    fig.update_layout(title=' Global flow data over all stations',xaxis_title='SZA',yaxis_title='Irradiance [W/m^2]')
    
    # Show the plot
    fig.show()
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
    
#############################################################################################################
def plot_scatter(df, x_col, y_col):
    """
    Trace un nuage de points à partir d'une DataFrame.
    
    Args:
        df (pandas.DataFrame): La DataFrame contenant les données.
        x_col (str): Le nom de la colonne pour l'axe des abscisses.
        y_col (str): Le nom de la colonne pour l'axe des ordonnées.
    Return:
        none
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col], alpha=1, s=0.00005, label ='Values')
    plt.scatter(df[x_col], df['physical_limit_ghi'], alpha=1, s=0.00005,label ='BSRN limit')
    plt.title('Physical limits and BRSN criteria')
    plt.xlabel(x_col+'(Solar zenith angle)')
    plt.ylabel(f'{y_col.upper()} [$W/m^2$]')
    plt.xlim((0,90))
    plt.grid(False)
    plt.legend()
    plt.show()

#############################################################################################################
def save_dni_estimation(path, name):
    """
    Save DNI estimation in .csv file
    
    Args:
        path (str) : path of .csv files
        name (str) : name of station
    Return:
        none

    Example : 
        path = 'data_brute/south_africa/durban_irrad.csv'
        name = 'durban'
    """
    try:
        df = pd.read_csv(path, sep=';', index_col=0)
        good_df = one_column_ghi_dhi(df)
        dni_estimation = estimation_dni_physical_limits(good_df, data_geo, name , 'Indian/Reunion')
        dni_estimation.to_csv(f'flow_all_data/{name}.csv', sep=';', index=True)
        return "Backup Success"
    except Exception as e:
        print(f"Error: {e}")
        return "Error"

#############################################################################################################
def save_flow_data(data_geo,chemins):
    """
    Save data in csv files

    Args:
        data_geo (dataframe): dataframe contains all name of stations SWOI.
        chemins (list): list contains all path file.
        
    Returns:
        None

    Example:
        data_geo = pd.read_csv('data_geo_all_station.csv', sep=';', index_col=0)
        chemins  = ['data_brute/south_africa/durban_irrad.csv',...,'data_brute/comores/hahaya_irrad.csv']
    """
    stations_name = data_geo.index.tolist()
    for i in range(len(stations_name)):
        print('--> '+ stations_name[i].upper())
        calculer_temps_execution(save_dni_estimation, chemins[i] ,stations_name[i])
    return 

#############################################################################################################
def all_data(data_geo):
    stations_name = data_geo.index.tolist()
    df_concatene  = pd.DataFrame()
    for name in stations_name:
        print(name)
        df_station = pd.read_csv(f'flow_all_data/{name}.csv', sep=';', index_col=0)
        df_concatene = pd.concat([df_concatene, df_station])
    return df_concatene

#############################################################################################################
def plot_scatter(df, x_col, y_col,name_station,path,size):
    """
    Trace un nuage de points à partir d'une DataFrame.
    
    Args:
        df (pandas.DataFrame): La DataFrame contenant les données.
        x_col (str): Le nom de la colonne pour l'axe des abscisses.
        y_col (str): Le nom de la colonne pour l'axe des ordonnées.
    """

    if y_col == 'ghi':
        plt.figure(figsize=(8, 6))
        plt.scatter(df[x_col], df[y_col], alpha=1, s=size, label='Values',color = 'black')
        plt.scatter(df[x_col], df['physical_limit_ghi'], alpha=1, s=0.5, label='BSRN limit GHI',color = 'red')
        custom_lines = [Line2D([0], [0], color='black', lw=1),Line2D([0], [0], color='red', lw=1)]
        plt.title(f'Physical limits and BRSN criteria ({name_station})')
        plt.xlabel(x_col + ' (Solar zenith angle)')
        plt.ylabel(f'{y_col.upper()} [$W/m^2$]')

        # Limite de l'axe des ordonnées à 2500 si nécessaire
        max_value = np.max(df[y_col])
        min_value = np.min(df[y_col])
        if max_value > 4000:
            plt.ylim(min_value-100, 4000 +100)
            
        plt.grid(False)
        plt.legend(custom_lines, ['Values of '+y_col.upper(), 'BSRN limit '+y_col.upper()],loc='upper right')
        # Sauvegarde du plot en PNG
        plt.savefig(path+f'ghi/{name_station}.png', bbox_inches='tight', pad_inches=0.05)
        
        plt.show()
    elif y_col == 'dhi':
        plt.figure(figsize=(8, 6))
        plt.scatter(df[x_col], df[y_col], alpha=1, s=size, label='Values',color = 'black')
        plt.scatter(df[x_col], df['physical_limit_dhi'], alpha=1, s=0.5, label='BSRN limit DHI',color = 'red')
        custom_lines = [Line2D([0], [0], color='black', lw=1), Line2D([0], [0], color='red', lw=1)]
        plt.title(f'Physical limits and BRSN criteria ({name_station})')
        plt.xlabel(x_col + ' (Solar zenith angle)')
        plt.ylabel(f'{y_col.upper()} [$W/m^2$]')
        
        plt.grid(False)
        plt.legend(custom_lines, ['Values of '+y_col.upper(), 'BSRN limit '+y_col.upper()],loc='upper right')
        # Sauvegarde du plot en PNG
        plt.savefig(path+f'dhi/{name_station}.png', bbox_inches='tight', pad_inches=0.05)
        
        plt.show()
    else:
        plt.figure(figsize=(8, 6))
        plt.scatter(df[x_col], df[y_col], alpha=1, s=size, label='Values',color = 'black')
        plt.scatter(df[x_col], df['physical_limit_dni'], alpha=1, s=0.5, label='BSRN limit DNI',color = 'red')
        custom_lines = [Line2D([0], [0], color='black', lw=1), Line2D([0], [0], color='red', lw=1)]
        plt.title(f'Physical limits and BRSN criteria ({name_station})')
        plt.xlabel(x_col + ' (Solar zenith angle)')
        plt.ylabel(f'{y_col.upper()} [$W/m^2$]')

        # Limite de l'axe des ordonnées à 2500 si nécessaire
        max_value = np.max(df[y_col])
        min_value = np.min(df[y_col])
        if max_value > 2500:
            plt.ylim(min_value-100, 2500 +100)
            
        plt.grid(False)
        plt.legend(custom_lines, ['Values of '+y_col.upper(), 'BSRN limit '+y_col.upper()],loc='upper right')
        # Sauvegarde du plot en PNG
        plt.savefig(path+f'dni/{name_station}.png', bbox_inches='tight', pad_inches=0.05)
        
        plt.show()
    return

#############################################################################################################
def show_flow_data_all_stations():
    # is necceserely to import os 
    liste_fichiers = glob.glob('flow_all_data/' + '*.csv')
    noms_fichiers_sans_extension = [os.path.splitext(chemin.split('\\')[-1])[0] for chemin in liste_fichiers]
    for i in range(len(noms_fichiers)):
        df = pd.read_csv(liste_fichiers[i], sep=';', index_col=0)
        df = df.drop(df[df['SZA'] > 90].index)
        calculer_temps_execution(plot_scatter,df, 'SZA', 'ghi',noms_fichiers_sans_extension[i],'img/flow_data/',0.0001)
        calculer_temps_execution(plot_scatter,df, 'SZA', 'dhi',noms_fichiers_sans_extension[i],'img/flow_data/',0.0001)
        calculer_temps_execution(plot_scatter,df, 'SZA', 'dni',noms_fichiers_sans_extension[i],'img/flow_data/',0.0001)
    return

#############################################################################################################
def show_flow_data_all_stations_useful():
    # is necceserely to import os 
    liste_fichiers = glob.glob('flow_data_useful_limit/' + '*.csv')
    noms_fichiers_sans_extension = [os.path.splitext(chemin.split('\\')[-1])[0] for chemin in liste_fichiers]
    for i in range(len(noms_fichiers_sans_extension)):
        df = pd.read_csv(liste_fichiers[i], sep=';', index_col=0)
        df = df.drop(df[df['SZA'] > 90].index)
        calculer_temps_execution(plot_scatter,df, 'SZA', 'ghi',noms_fichiers_sans_extension[i],'img/flow_data_useful/',0.5)
        calculer_temps_execution(plot_scatter,df, 'SZA', 'dhi',noms_fichiers_sans_extension[i],'img/flow_data_useful/',0.5)
        calculer_temps_execution(plot_scatter,df, 'SZA', 'dni',noms_fichiers_sans_extension[i],'img/flow_data_useful/',0.5)
    return

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
def mae_of_list(y_true, y_pred):
    '''
    Calcule MAE of 2 list

    Args:
        y_true (lst) : list contaning in-situe data
        y_pred (lst) : list contaning SARAH-3 data
        
    Returns:
        mae (numpy.float64) : MAE value 

    Example:
        y_true = [1,52,652,61]
        y_pred = [-1,52,52,61]
    '''
    y_true_useful = np.array(y_true)
    y_pred_useful = np.array(y_pred)
    mae = np.nanmean(np.abs(y_pred_useful - y_true_useful)) # np.nanmean permit to calcul mean with np.nan values consideration
    return mae

#############################################################################################################
def mbe_of_list(y_true, y_pred):
    '''
    Calcule MAE of 2 list

    Args:
        y_true (lst) : list contaning in-situe data
        y_pred (lst) : list contaning SARAH-3 data
        
    Returns:
        mae (numpy.float64) : MAE value 

    Example:
        y_true = [1,52,652,61]
        y_pred = [-1,52,52,61]
    '''
    y_true_useful = np.array(y_true)
    y_pred_useful = np.array(y_pred)
    mbe = np.nanmean(y_pred_useful - y_true_useful) # np.nanmean permit to calcul mean with np.nan values consideration
    return mbe


#############################################################################################################
def rmse_of_list(y_true, y_pred):
    '''
    Calcule MAE of 2 list

    Args:
        y_true (lst) : list contaning in-situe data
        y_pred (lst) : list contaning SARAH-3 data
        
    Returns:
        mae (numpy.float64) : MAE value 

    Example:
        y_true = [1,52,652,61]
        y_pred = [-1,52,52,61]
    '''
    y_true_useful = np.array(y_true)
    y_pred_useful = np.array(y_pred)
    
    value = np.nanmean((y_pred_useful - y_true_useful)**2) # np.nanmean permit to calcul mean with np.nan values consideration
    rmse = np.sqrt(value)
    return rmse

#############################################################################################################
def statistic_comparison_station(data_sarah,data_in_situ,data_geo):
    '''
    Creat a Dataframe with statistic indicator between data_sarah and data_in_situ

    Args:
        data_sarah (dataframe) : dataframe contaning sarah data
        data_in_situ (dataframe) : dataframe contaning in-situ data
        data_geo (dataframe) : dataframe contaning geographical data
        
    Returns:
        df_stat (dataframe) : dataframe contaning statistic indicator between data_sarah and data_in_situ
    Example:
        data_sarah = pd.read_csv('comparison/daily_mean/all_SARAH_data.csv' , sep=';', index_col=0)
        data_in_situ = pd.read_csv('comparison/daily_mean/all_in_situe_data.csv' , sep=';', index_col=0)
        data_geo = pd.read_csv('data_geo_adaptation.csv', sep=';', index_col=0)
    '''
    df_stat = pd.DataFrame(index=data_geo.index)
    list_mae = []
    list_mbe = []
    list_rmse = []
    for i in range(data_in_situ.shape[1]):
        list = data_in_situ.iloc[:,i].tolist()
        list_sarah = data_sarah.iloc[:,i].tolist()
        MAE = mae_of_list(list, list_sarah)
        MBE = mbe_of_list(list, list_sarah)
        RMSE = rmse_of_list(list, list_sarah)
        list_mae.append(MAE)
        list_mbe.append(MBE)
        list_rmse.append(RMSE)
    df_stat['MAE'] = list_mae
    df_stat['MBE'] = list_mbe
    df_stat['RMSE'] = list_rmse
    df_stat['Altitude'] = data_geo['Altitude'].tolist()


    df_stat.name = 'Statistic comparison station'
    return df_stat



#############################################################################################################
def remove_extrem_value(data):
    '''
    This function permit to remove extrem value on dataframe 

    Args:
        data (dataframe) : dataframe contaning all data por one station
        
    Returns:
        df (dataframe) : dataframe contaning all data por one station filter
    Example:
        data = pd.read_csv('data_urmoufia/urmoufia.csv', sep=';', index_col=0)
    '''
    init_value = count_nan(data)
    df=data.copy()
    # filter of maximum value
    filtre_ghi_max = df['ghi'] >  df['physical_limit_ghi']
    filtre_dhi_max = df['dhi'] >  df['physical_limit_dhi']
    filtre_dni_ground_max = df['dni_ground'] >  df['physical_limit_dni']
    filtre_dni_max = df['dni'] >  df['physical_limit_dni']

    # filter of  minimum value
    filtre_ghi_min = df['ghi'] < -4
    filtre_dhi_min = df['dhi'] < -4
    filtre_dni_ground_min = df['dni_ground'] < -4
    filtre_dni_min = df['dni'] < -4

    df.loc[filtre_ghi_max, 'ghi'] = np.nan
    df.loc[filtre_dhi_max, 'dhi'] = np.nan
    df.loc[filtre_dni_ground_max, 'dni_ground'] = np.nan
    df.loc[filtre_dni_max, 'dni'] = np.nan

    df.loc[filtre_ghi_min, 'ghi'] = np.nan
    df.loc[filtre_dhi_min, 'dhi'] = np.nan
    df.loc[filtre_dni_ground_min, 'dni_ground'] = np.nan
    df.loc[filtre_dni_min, 'dni'] = np.nan

    final_value = count_nan(df)

    print(f'Remove values : {final_value-init_value}')
    return df


#############################################################################################################
def compare_data_old_new(df_new,df_old):
    '''
    This function permit to compare old and new data in the same data station

    Args:
        df_new (dataframe) : dataframe contaning new data
        df_old (dataframe) : dataframe contaning old data
        
    Returns:
        None
        
    Example:
        if we are : df_urbsrn_dni = pd.read_csv('data_urmoufia/urbsrn.csv', sep=';', index_col=0)
                    df_oldbsrn_dni = pd.read_csv('data_urmoufia/oldbsrn.csv', sep=';', index_col=0)
                    df_urbsrn_dni_filter = remove_extrem_value(df_urbsrn_dni)
                    df_oldbsrn_dni_filter = remove_extrem_value(df_oldbsrn_dni)
                    start_index=df_urbsrn_dni_filter.index.min()
                    end_index=df_oldbsrn_dni_filter.index.max()

        then we take :  df_new = df_urbsrn_dni_filter[start_index:end_index]
                        df_old = df_oldbsrn_dni_filter[start_index:end_index]
            
    '''

    # Création de la figure et des sous-graphiques
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))
    var = ['ghi','dhi','dni','dni_ground']
    color_name = ['blue','red','green','magenta']

    for i in range(len(var)):
        min_value_new =  df_new[var[i]].min()
        max_value_new =  df_new[var[i]].max()

        min_value_old =  df_old[var[i]].min()
        max_value_old =  df_old[var[i]].max()

        valeur_min = min(min_value_new, min_value_old)
        valeur_max = max(max_value_new, max_value_old)
        
        # Ajouter la ligne où x = y
        x_values = np.linspace(valeur_min, valeur_max, 100)  # Créer des valeurs de x

        list_old = df_old[var[i]].tolist()
        list_new = df_new[var[i]].tolist()

        RMSE_value = round(rmse_of_list(list_new, list_old),2)

        axs[i].plot(x_values, x_values, color='black', linestyle='--',linewidth=1, label='x=y')
        axs[i].scatter(df_old[var[i]],df_new[var[i]], color=color_name[i],s=0.5,label='value')
        axs[i].set_title(var[i].upper())
        axs[i].set_xlabel('data_oldbsrn')
        axs[i].set_ylabel('data_urbsrn')
        axs[i].legend(loc='lower right')
        axs[i].text(1.1, 0.5, f'RMSE = {RMSE_value}', transform=axs[i].transAxes, fontsize=12, verticalalignment='center')
    
    # Ajustement de l'espacement entre les sous-graphiques
    plt.tight_layout()
    
    # Affichage de la figure
    plt.show()
    return


#############################################################################################################
def remove_nan_values(arr):
    """
    Remove NaN values from a NumPy array.

    Parameters:
        arr (numpy.ndarray): Input array containing NaN values.

    Returns:
        numpy.ndarray: Array without NaN values.
    """
    # Identifier les indices des valeurs NaN
    nan_indices = np.isnan(arr)

    # Supprimer les valeurs NaN en indexant le tableau avec les indices inversés
    arr_without_nan = arr[~nan_indices]

    return arr_without_nan

#############################################################################################################
def convert_netCDF_to_df(path_of_netCDF_file,variable):
    '''
    This function permit to remove extrem value on dataframe 

    Args:
        path_of_netCDF_file (lst) : list contains all netCDF name file
        variable (str) : name of variable
        
    Returns:
        df (dataframe) : dataframe contains all irradiance data of SARAH
    Example:
        path_of_netCDF_file = ['data_sarah3/DNI/instantaneous/netCDF\\DNIin200812010000004UD1000101UD.nc',
                               'data_sarah3/DNI/instantaneous/netCDF\\DNIin200812020000004UD1000101UD.nc']
        variable = 'SIS'
    '''
    df = pd.DataFrame()
    k=0
    for path in path_of_netCDF_file:
        k=k+1
        print(f'\r{k}', end='', flush=True)
        # Charger le fichier netCDF
        dataset = xr.open_dataset(path)
        
        # Extraire les données de DNI et le timestamp
        irrad_data = dataset[variable]  # Remplacez 'DNI' par le nom exact de la variable dans votre fichier netCDF
        
        # Convertir les données en DataFrame
        df_irrad = irrad_data.to_dataframe()
        
        # Réinitialiser l'index pour obtenir une colonne timestamp
        df_irrad.reset_index(inplace=True)
        df_irrad = df_irrad.drop(['lat', 'lon'], axis=1)
        
        # Convertir le temps en index et le mettre au format datetime
        df_irrad['time'] = pd.to_datetime(df_irrad['time'])
        df_irrad.set_index('time', inplace=True)
        
        # Renommer la colonne 'A' en 'Alpha'
        df_irrad.rename(columns={variable: variable+'_SARAH_3'}, inplace=True)
        df_irrad.replace(-999, np.nan, inplace=True)
        df = pd.concat([df,df_irrad])
        
    print()
    df.to_csv(f'data_urmoufia/data_UTC/SARAH_3_{variable}_moufia.csv', sep=';', index=True)
    return df
