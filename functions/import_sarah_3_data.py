# Libraries 
import glob
import os
import seaborn as sns
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
import warnings
import plotly.express as px
from plotly.subplots import make_subplots



sns.set(style="darkgrid", palette="bright",context = "talk",font="fantasy" )

# Désactiver l'affichage des FutureWarnings temporairement
warnings.simplefilter(action='ignore', category=FutureWarning)


def nc_to_dataframe(file_path):
    """
    Convertit un fichier .nc (NetCDF) en DataFrame.
    
    Args:
    - file_path (str): Chemin vers le fichier .nc
    
    Returns:
    - DataFrame: DataFrame contenant les données du fichier .nc
    """
    # Charger les données NetCDF en tant que dataset
    dataset = xr.open_dataset(file_path)
    
    # Convertir le dataset en DataFrame
    df = dataset.to_dataframe().reset_index()
    
    return df



def hourly_mean(df):
    df.index = pd.to_datetime(df.index)
    data = df.copy()
    # Group data by hour and calculate the mean
    df_hourly_mean = data.resample('H').mean()
    return df_hourly_mean

def daily_mean(df):
    df.index = pd.to_datetime(df.index)
    data = df.copy()
    # Group data by hour and calculate the mean
    df_daily_mean = data.resample('D').mean()
    return df_daily_mean

def monthly_mean(df):
    df.index = pd.to_datetime(df.index)
    data = df.copy()
    # Group data by hour and calculate the mean
    df_mounthly_mean = data.resample('M').mean()
    return df_mounthly_mean

def yearly_mean(df):
    df.index = pd.to_datetime(df.index)
    data = df.copy()
    # Group data by hour and calculate the mean
    df_yearly_mean = data.resample('Y').mean()
    return df_yearly_mean


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
        map = ccrs.PlateCarree()
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


def regroup_sarah_data_mounthly_mean(liste_of_file_path,data_geo):
    """
    Save the pixel value on dataframe with SARAH-3 data at netCDF files

    Args:
        liste_of_file_path (lst) : liste of path data
        data_geo (dataFrame) : dataframe which contains all geographical data of stations

    Returns:
        data_geo (dataFrame) : dataframe which contains all geographical data of stations

    Example :
        liste_of_file_path = ["data_sarah3/2020/DNImm202001010000004UD1000101UD.nc",....,"data_sarah3/2020/DNImm202012010000004UD1000101UD.nc",]
        data_geo = pd.read_csv('data_geo_adaptation.csv', sep=';', index_col=0)
    """

    # make a copy to be able to manipulate the dataframe without modifying the original
    data_modif = data_geo.copy()
    index_data_geo = data_geo.index.tolist()
    df = pd.DataFrame(columns=index_data_geo)
    for j in range(len(liste_of_file_path)):
        url = liste_of_file_path[j]
        ncfile = nc.Dataset(url)
        STANDARD_NAME = "surface_direct_along_beam_shortwave_flux_in_air" #DNI
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
        
        # Converting time to human-readable format
        time_array = np.array([datetime.fromtimestamp(t) for t in time_unix])
        dni_value_sarah = var_array[0]
        dni_liste = []
        liste_latitude = list(latitude.data)
        liste_longitude = list(longitude.data)
        for i in range(data_modif.shape[0]):
            float64_number_1 = data_modif['Latitude_modif'][i]
            float32_number_1 = float64_number_1.astype(np.float32)
            float64_number_2 = data_modif['Longitude_modif'][i]
            float32_number_2 = float64_number_2.astype(np.float32)
            indice_lat = liste_latitude.index(float32_number_1)
            indice_long = liste_longitude.index(float32_number_2)
            val = dni_value_sarah[indice_lat][indice_long]
            dni_liste.append(val)
        dni_value = dni_liste
        index = pd.to_datetime(ncfile.time_coverage_start)
        df.loc[index] = dni_value
    # Renommer toutes les colonnes
    new_column_names = [col + '_DNI_SARAH_3' for col in df.columns]
    # Renommer les colonnes
    df.rename(columns=dict(zip(df.columns, new_column_names)), inplace=True)
    df.name = 'DNI_SARAH_3'
    return df

def show_irrad_all_stations(df_data):
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

    # Ajouter chaque colonne à la figure
    for column in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=column,line=dict(width=1)))
    
    
    # Customize layout
    fig.update_layout(title=f'{df_data.name}',xaxis_title='Time [UTC+4]',yaxis_title='Irradiance [W/m^2]')
    
    # Show the plot
    fig.show()
    return

def compar_data(liste_of_data, stations_names, start_date , end_date , function_mean, variable):
    """
    Regroup in-situe data in one dataframe to compare with SARAH data

    Args:
        liste_of_data (lst): liste of dataFrame containing irradiance data in-situe.
        stations_names (lst): liste of in-situe name stations.
        start (str) : start time of comparison.
        end (str) : finish time of comparison.
        function_mean (function): mean function you use.
        variable (str) : the name of comparative variable.
        
    Returns:
        df (dataframe) : dataframe contains all data stations.
        
    Example:
        liste_of_useful_data = [df_durban_irrad_useful,...,df_hahaya_irrad_useful]
        stations_names = ['durban',...,'hahaya']
        stations_names = data_geo.index.tolist()
        function_mean = mounthly_mean
        start_date = '2008-12-01 00:00:00'
        end_date = '2024-04-01 00:00:00'
        variable = 'dni'
    """
    L = len(liste_of_data)
    df = pd.DataFrame()
    for i in range(L):
        print(f'{i+1}/{L}') # turn indicator 
        df_1 = liste_of_data[i]
        df_1.index = pd.to_datetime(df_1.index) # conversion of index
        # Créer un nouvel index avec la plage de dates souhaitée
        index_before = pd.date_range(start=start_date, end=df_1.index[0].strftime('%Y-%m-%d %H:%M:%S'), freq='H', tz='UTC+04:00')
        index_after = pd.date_range(start=df_1.index[-1].strftime('%Y-%m-%d %H:%M:%S'), end=end_date, freq='H', tz='UTC+04:00')
    
        combined_index_1 = index_before.append(df_1.index)
        combined_index_2 = combined_index_1.append(index_after)
        df_1 = df_1.reindex(combined_index_2)
        mean_data = df_1.apply(function_mean) # this row permit to apply the mean on the data
        #select_data = mean_data[start:end] # this row permit to select the time zone useful
        df[stations_names[i]+f'_{variable.upper()}_in_situe'] = mean_data[variable] # this row permit to upload each data stations in new dataframe
    #df.index = pd.to_datetime(df.index) # conversion of index
    # Définir le nom du DataFrame
    df.name = variable.upper()+'_in_situe'
    return df

def show_diff_2_data(data_sarah,data_in_situe):
    """
    Show mean value of irradiance data for each SWOI stations with Scatter plot (in-situe and SARAH)

    Args:
        data_sarah (DataFrame): DataFrame containing mean irradiance data of SARAH.
        data_in_situe (DataFrame): DataFrame containing mean irradiance data in-situe.
        
    Returns:
        None
        
    Example:
        data_sarah = regroup_sarah_data_mounthly_mean(liste_of_file_path,data_geo)
        data_in_situe = compar_data(liste_of_data, stations_names, start , end, function_mean, variable)
    """
    df = data_sarah.copy()
    df_1 = data_in_situe.copy()
    df_1.index = df.index

    # Create the figure
    fig = go.Figure(layout=dict(width=1000, height=600))

    # Ajouter chaque colonne à la figure
    for column in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=column,line=dict(width=1,color='red')))
    # Ajouter chaque colonne à la figure
    for column in df_1.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df_1[column], mode='lines', name=column,line=dict(width=1,color='blue')))
        #fig.add_trace(go.Scatter(x=df.index, y=df_1[column], mode='lines', name=column,line=dict(width=1, dash='dash'),color='blue'))

    # Customize layout
    fig.update_layout(title='GHI, DHI and DNI',xaxis_title='Time [UTC+4]',yaxis_title='Irradiance [W/m^2]')
    
    # Show the plot
    fig.show()
    return


def show_irrad_all_stations_Heatmap(df_data):
    """
    Show mean value of irradiance data for each SWOI stations

    Args:
        df_data (DataFrame): DataFrame containing mean irradiance data.
        
    Returns:
        None
    """
    df = df_data.copy()

    # Transposer le DataFrame
    df_transposed = df.transpose()

    # Create the figure
    fig = go.Figure(layout=dict(width=1000, height=600))

    # Create colormesh
    fig.add_trace(go.Heatmap(z=df_transposed.values, x=df_transposed.columns, y=df_transposed.index,  colorscale='Viridis'))

    # Customize layout
    fig.update_layout(title=f'{df_data.name}', xaxis_title='Time [UTC+4]', yaxis_title='Irradiance [W/m^2]')

    # Set y-axis tickmode to 'linear' and specify dtick
    fig.update_yaxes(tickmode='linear', dtick=1)  # Set the dtick value to adjust spacing between elements

    # Show the plot
    fig.show()

    return


def show_irrad_all_stations_Heatmap_2_data(df_sarah, df_in_situe):
    """
    Show mean value of irradiance data for each SWOI stations with Heatmap plot

    Args:
        df_sarah (DataFrame): DataFrame containing mean irradiance data of SARAH.
        df_in_situe (DataFrame): DataFrame containing mean irradiance data in-situe.
        
    Returns:
        None

    Example:
        data_sarah = regroup_sarah_data_mounthly_mean(liste_of_file_path,data_geo)
        data_in_situe = compar_data(liste_of_data, stations_names, start , end, function_mean, variable)
    """
    df1 = df_sarah.copy()
    df2 = df_in_situe.copy()
    df1.index = df2.index

    # Transposer les DataFrames
    df_transposed1 = df1.transpose()
    df_transposed2 = df2.transpose()

    # Créer les colormesh pour chaque DataFrame
    trace1 = go.Heatmap(z=df_transposed1.values, x=df_transposed1.columns, y=df_transposed1.index, colorscale='Agsunset')
    trace2 = go.Heatmap(z=df_transposed2.values, x=df_transposed2.columns, y=df_transposed2.index, colorscale='Agsunset')

    # Créer une disposition avec des subplots empilés verticalement
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=[df_sarah.name, df_in_situe.name])

    # Ajouter les colormesh aux subplots
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=2, col=1)

    # Personnaliser la mise en page de la figure
    fig.update_layout(title="Irradiance Comparison", height=1200, showlegend=False, paper_bgcolor='white', xaxis=dict(showgrid=True), yaxis=dict(showgrid=False))

    # Définir les titres des axes pour chaque subplot
    fig.update_yaxes(title_text="Stations SWOI", row=1, col=1)
    fig.update_yaxes(title_text="Stations SWOI", row=2, col=1)
    fig.update_xaxes(title_text="Time [UTC+4]", row=2, col=1)

    # Set y-axis tickmode to 'linear' and specify dtick
    fig.update_yaxes(tickmode='linear', dtick=1)  # Set the dtick value to adjust spacing between elements
    # Adjust font size of y-axis tick labels
    fig.update_yaxes(tickfont=dict(size=10))  # Adjust the size value according to your preference


    fig.show()

    return


def calculer_temps_execution(fonction, *args, **kwargs):
    debut = time.time()
    resultat = fonction(*args, **kwargs)
    fin = time.time()
    temps_execution = fin - debut
    print(f"Temps d'exécution de {fonction.__name__}: {temps_execution} secondes")
    return resultat


def regroup_sarah_data_daily_mean_par_station(liste_of_file_path,data_geo,name_station):
    """
    Save the pixel value on dataframe with SARAH-3 data at csv files

    Args:
        liste_of_file_path (lst) : liste of path data
        data_geo (dataFrame) : dataframe which contains all geographical data of stations

    Returns:
        data_geo (dataFrame) : dataframe which contains all geographical data of stations

    Example :
        liste_of_file_path = ["data_sarah3/2020/DNImm202001010000004UD1000101UD.nc",....,"data_sarah3/2020/DNImm202012010000004UD1000101UD.nc",]
        data_geo = pd.read_csv('data_geo_adaptation.csv', sep=';', index_col=0)
    """

    # make a copy to be able to manipulate the dataframe without modifying the original
    data_modif = data_geo.copy()
    index_data_geo = data_geo.index.tolist()
    df = pd.DataFrame(columns=[name_station])
    number_index = index_data_geo.index(name_station)
    for j in range(len(liste_of_file_path)):
        url = liste_of_file_path[j]
        ncfile = nc.Dataset(url)
        STANDARD_NAME = "surface_direct_along_beam_shortwave_flux_in_air" #DNI
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
        
        # Converting time to human-readable format
        time_array = np.array([datetime.fromtimestamp(t) for t in time_unix])
        dni_value_sarah = var_array[0]
        dni_liste = []
        liste_latitude = list(latitude.data)
        liste_longitude = list(longitude.data)
    
        float64_number_1 = data_modif['Latitude_modif'][number_index]
        float32_number_1 = float64_number_1.astype(np.float32)
        float64_number_2 = data_modif['Longitude_modif'][number_index]
        float32_number_2 = float64_number_2.astype(np.float32)
        #indice_lat = liste_latitude.index(round(float(float32_number_1),3))
        #indice_long = liste_longitude.index(round(float(float32_number_2),3))

        indice_lat = liste_latitude.index(float32_number_1)
        indice_long = liste_longitude.index(float32_number_2)
    
    
        
        #indice_lat = liste_latitude.index(float32_number_1)
        #indice_long = liste_longitude.index(float32_number_2)
        val = dni_value_sarah[indice_lat][indice_long]
        dni_liste.append(val)
        dni_value = dni_liste
        
        index = pd.to_datetime(ncfile.time_coverage_start)
        df.loc[index] = dni_value
        ncfile.close()
    # Renommer toutes les colonnes
    new_column_names = [col + '_DNI_SARAH_3' for col in df.columns]
    # Renommer les colonnes
    df.rename(columns=dict(zip(df.columns, new_column_names)), inplace=True)
    #df.name = name_station
    # Remplacer -999 par np.nan dans tout le DataFrame
    df.replace(-999, np.nan, inplace=True)
    df = round(df)
    df.to_csv(f'data_sarah3/DNI/daily/dataframe/{name_station}.csv', sep=';', index=True)
    return df

