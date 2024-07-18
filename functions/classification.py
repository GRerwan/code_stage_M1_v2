import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
import time
import pvlib
from pvlib.location import Location
import caelus
from matplotlib.ticker import FixedLocator, FixedFormatter
#############################################################################################################
#sns.set(style="darkgrid", palette="bright",context = "paper",font="Arial" )
sns.set(style="darkgrid", palette="bright",context = "talk",font="Arial" )
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


def formalisation_data(data_filter,var):
    mydf = data_filter.copy()
    # Supprimer le nom de l'index de la copie
    mydf.index.name = None
    mydf.index = pd.to_datetime(mydf.index)
    mydf = mydf.reset_index()
    mydf = mydf.rename(columns={'index': 'timestamp'})
    #mydf = mydf.rename(columns={variable: 'SIS_filtred'})
    mydf['date'] = mydf['timestamp'].dt.strftime('%Y/%m/%d')
    mydf['date'] = pd.to_datetime(mydf['date'], format='%Y/%m/%d')
    mydf['heure'] = mydf['timestamp'].dt.strftime('%H:%M:%S')
    mydf['ID'] =  ['D{}'.format(i + 1) for i in mydf.index]
    # Trouver la première date
    first_date = mydf['date'].min()
    # Calculer le nombre de jours écoulés depuis la première date
    mydf['ID_date'] = (mydf['date'] - first_date).dt.days + 1
    return round(mydf,2)

def formalisation_data_bis(data_filter):
    # Supprimer le nom de l'index et le convertir en datetime
    mydf = data_filter.copy()
    #mydf = mydf.round(3)
    mydf.index.name = None
    mydf.index = pd.to_datetime(mydf.index)
    
    # Reset index and rename the new column to 'timestamp'
    mydf.reset_index(inplace=True)
    mydf.rename(columns={'index': 'timestamp'}, inplace=True)
    
    # Create 'date' and 'heure' columns
    mydf['date'] = mydf['timestamp'].dt.normalize()  # Faster than strftime and to_datetime
    mydf['heure'] = mydf['timestamp'].dt.strftime('%H:%M:%S')
    
    # Round the values and create 'ID' and 'ID_date' columns
    mydf['ID'] = ['D{}'.format(i + 1) for i in range(len(mydf))]
    first_date = mydf['date'].min()
    mydf['ID_date'] = (mydf['date'] - first_date).dt.days + 1
    
    return mydf
#############################################################################################################
def pivot_data(mydf,var):
    mydf2 = mydf.pivot_table(values=var,index='heure',columns='ID_date')
    return mydf2

#############################################################################################################
def raw_data_to_sky_type_ineichen_bis(raw_data, data_geograph, name_station):
    run_bsrn_data = raw_data.copy()
    #run_bsrn_data = minute_mean(run_bsrn_data)
    # Optimiser la conversion de l'index en datetime directement au chargement du CSV
    run_bsrn_data.index = pd.to_datetime(run_bsrn_data.index)
    
    # Récupérer les informations géographiques
    info_geo = data_geograph.loc[name_station]
    
    # Localisation
    loc = Location(latitude=info_geo.Latitude, longitude=info_geo.Longitude, tz='UTC', altitude=info_geo.Altitude)
    
    # Calcul de la position solaire
    sp = loc.get_solarposition(run_bsrn_data.index, temperature=25)
    
    # Calcul de l'airmass relatif et absolu
    r_a = pvlib.atmosphere.get_relative_airmass(sp['zenith'])
    a_a = pvlib.atmosphere.get_absolute_airmass(r_a)
    
    # Turbidité Linke
    #linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(run_bsrn_data.index, loc.latitude, loc.longitude)
    
    
    # Calcul du ciel clair (clear sky)
    cs = loc.get_clearsky(run_bsrn_data.index, model='ineichen', linke_turbidity=2)
    
    # Ajout des colonnes directement dans le DataFrame original pour éviter les copies intermédiaires
    run_bsrn_data['sza'] = run_bsrn_data['SZA']
    run_bsrn_data['eth'] = run_bsrn_data['extra_radiation']
    run_bsrn_data['longitude'] = info_geo.Longitude
    run_bsrn_data['ghics'] = cs['ghi'].values
    run_bsrn_data['ghicda'] = cs['ghi'].values
    
    # Sélection des colonnes finales pour créer data_to_clas
    data_to_clas = run_bsrn_data[['longitude', 'sza', 'eth', 'ghi', 'ghics', 'ghicda']].copy()
    #print('Step 1/3 : Data is ready to classifier')

    sky_type = caelus.classify(data_to_clas,full_output =True)
    #print('Step 2/3 : Data is classifier')
    data_to_clas['sky_type'] = sky_type.sky_type.values
    data_to_clas = data_to_clas.round(3)
    
    #good_data = formalisation_data_bis(data_to_clas)
    #print('Step 3/3 : Data is on  good format')


    return data_to_clas



def list_classification_per_station(good_data):
    """
    Réorganisation de la Dataframe avec en créant une dataframe par classification
    
    Args:
        good_data (dataframe) : dataframe contains irradiance and type sky
    Return:
        dfs_formate (list) : 

    Example : 
        good_data = formalisation_data_bis(raw_data_to_sky_type_ineichen_bis(raw_data, data_geograph, name_station))
    """

    # Grouper les données par 'sky_type'
    df = good_data.copy()
    grouped = df.groupby('sky_type')
    grouped
    
    # Créer un dictionnaire pour stocker les DataFrames
    dfs = [group for group in grouped]

    dfs_formate = []
    for i in range(len(dfs)):
        data_format = pivot_data(dfs[i][1],'ghi')
        dfs_formate.append(data_format)

    return dfs_formate
