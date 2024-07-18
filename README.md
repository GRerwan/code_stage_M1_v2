# Intership of Master Energy : Inter-comparison and validation against in-situ measurements of satellite estimates of incoming solar radiation for Indien Ocean

Ce projet Github est le développement que j'ai fait lors de mon stage de M1 ENERGIE, qui a pour objectif de pouvoir comparer et mettre en relation les enregistrements de données climatiques par satellite sur l'irradiation de la surface solaire (SARAH-3) avec les données de messures solaires au sol dans le réseaux IOS-net basé dans le Sud-Ouest de l'Océan Indien.

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
list_of_dat_per_country =liste_of_link(url)
#############################################################################################################
name_country = ['south_africa','seychelle','mauritius','mada','reunion','comores']
#############################################################################################################
def save_all_raw_data_to_csv_file(list_of_dat_per_country,name_country):
    L = len(name_country)
    for i in range(L):
        k = len(list_of_dat_per_country[i])
        for j in range(k):
            name_station , df_station = affiche_data_reunion(list_of_dat_per_country[i][j])
            print(name_station)
            df_station.to_csv(f'data_raw_UTC/{name_country[i]}/{name_station}_irrad.csv', sep=';', index=True) 
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
calculer_temps_execution(save_all_raw_data_to_csv_file,list_of_dat_per_country,name_country)
#############################################################################################################
```
+ La fonction `liste_of_link` permet de récupérer tout les liens `.nc` (netCDF files) disponible sur le site IOS-net [1], cette fonction renvoie donc une liste de 6 sous-liste avec chaque sous correspondant aux liens des respectifs aux 6 zones d'étues : Afrique du Sud, Seychelles, Mauritius, Madagascar, La Réunion et Comores.
+ La fonction `save_all_raw_data_to_csv_file` permet convertir toutes les liens `.nc` en dataframe, ensuite les dataframes sont converties en fichiers `.csv` puis sauvegarder localement.
+ La conversion des fichiers `.nc` en dataframe se fait grâce la fonction `nc_to_dataframe_second`, dans cette fonction le Timestamp de la dataframe est créé et uniquement les données d'irradiance sont conservé avec le code suivant :
```python
resultat_1 = [var for var in liste_variable if 'DHI' in var or 'GHI' in var or 'DNI' in var]
```
Ainsi, on obtient par exemple pour le site du Moufia ("urmoufia") un tableau comme ci-dessous : 

|                     |   GHI_pa01_Avg |   DHI_pa01_Avg |   GHI_qb01_Avg |   GHI_pa03_Avg |   DHI_pa03_Avg |   DHI_pw01_Avg |   GHI_pn03_Avg |   GHI_qb03_Avg |   DHI_pn03_Avg |   DHI_pw03_Avg |   DNI_px03_Avg |
|:--------------------|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|
| 2020-01-24 08:00:00 |            nan |            nan |            nan |            nan |            nan |            nan |          138   |          43.01 |          144.6 |          144.4 |          191.8 |
| 2020-01-24 08:01:00 |            nan |            nan |            nan |            nan |            nan |            nan |          140.6 |          43.77 |          147.5 |          145.6 |          192   |
| 2020-01-24 08:02:00 |            nan |            nan |            nan |            nan |            nan |            nan |          143.9 |          44.73 |          149.3 |          148.4 |          192   |
| 2020-01-24 08:03:00 |            nan |            nan |            nan |            nan |            nan |            nan |          150.4 |          46.1  |          156.8 |          152.4 |          191.7 |
| 2020-01-24 08:04:00 |            nan |            nan |            nan |            nan |            nan |            nan |          157.7 |          47.71 |          165.4 |          156.8 |          192.2 |

On voit ainsi que les données d'irradiance ne sont pas encore utilisable pour l'étude car en effet pour une même période deux valeurs de GHI ou de DHI peuvent être observé. Cela s'explique par le fait que "deux capteurs mesures en même temps le GHI ". Ainsi, afin de pouvoir utilisé les données de GHI, DHI et DNI il faut avoir une seule colonne de chaque irradiance. La fonction `one_column_ghi_dhi`permet de mettre les données dans une seule colonne de GHI, de DNI et de DHI : 

|                     |   ghi |   dhi |   dni_ground |
|:--------------------|------:|------:|-------------:|
| 2020-01-24 08:00:00 | 138   | 144.6 |        191.8 |
| 2020-01-24 08:01:00 | 140.6 | 147.5 |        192   |
| 2020-01-24 08:02:00 | 143.9 | 149.3 |        192   |
| 2020-01-24 08:03:00 | 150.4 | 156.8 |        191.7 |
| 2020-01-24 08:04:00 | 157.7 | 165.4 |        192.2 |

```python
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
```

### Estimate DNI with GHI and DHI

Pour le cas de la stations du moufia, il est possible d'avoir les données de DNI car cette station possède en un CMP22, cependant les autres stations du réseaux IOS-net possèdent seulement une SPN1 pour la messures de radiation solaire. Ainsi, il est important de pouvoir estimer le DNI pour les autres stations avec la fonction `estimation_dni_physical_limits` suivant qui permet d'une part d'estimer le DNI mais également permet d'avoir les limites physiques pour les irradiances : 
+ $-4 < GHI < S_a \times 1.5 \times \mu_0^{1.2} + 100$
+ $-4 < DHI < S_a \times 0.95 \times \mu_0^{1.2} + 50$
+ $-4 < DNI < S_a$

```python
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
```

Ensuite après avoir accès aux données complètes de l'irradiance, il est nécessaire d'appliquer le contrôle de qualité sur les données c'est-à-dire supprimer les données qui ne respect pas les limites physiques avec la fonction `quality_of_bsrn` suivant : 
```python
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
    df_minute_mean = df_1.resample('T').mean()

    # Returns the DataFrame with outliers replaced by np.nan
    return df_minute_mean
```

### Save raw data




### CLassification

Il savoir dans un premier temps que la classification qui sera abordé a été développé par **Jose A Ruiz-Arias** et que la classification dépend principalement du modèle de ciel utilisé, ainsi, un modèle de ciel précise permet d'ameliorer la précision pour la classification. Pour ma part dans ce stage pas manque de temps j'ai choisie un modèle de ciel clair relativement simple avec `pvlib`.

Les classifications qui sont disponible par le code de [Jose A Ruiz-Arias](https://github.com/jararias/caelus) : 

- OVERCAST
- THICK CLOUDS
- SCATTER CLOUDS
- THIN CLOUDS
- CLOUDLESS
- CLOUD ENHANCEMENT

+ Sud Ouest de l'Océan Indien

![Carte Sud Ouest de l'Océan Indian](https://github.com/user-attachments/assets/4ea890ad-d46d-4268-a860-ad59eeb3f520)

+ Sud de l'Afrique

![Durban](https://github.com/user-attachments/assets/9da9f121-1ce0-4015-99d3-458843c135a6)

+ Comores

![image](https://github.com/user-attachments/assets/f9af4b4b-7c4c-476c-9dd5-aadbe1dc842c)

+ Madagascar

![image](https://github.com/user-attachments/assets/6f5d309c-a810-4b72-9bf8-741e8ce68941)

+ Seychelles

![image](https://github.com/user-attachments/assets/baf6be9f-310f-4159-9335-a16f7c609321)

+ La Reunion

![image](https://github.com/user-attachments/assets/f4dd4835-9cd0-494a-8bc9-bda048f363ba)

+ Maurice

![image](https://github.com/user-attachments/assets/1738a24d-d771-418c-b865-5e1db431c392)
![image](https://github.com/user-attachments/assets/ccbd538f-89bb-4053-96ff-314c66d889a6)


**La taille des diagramme circulaire sont proportionnelles au valeurs moyenne du GHI au niveau de chaque stations.**





Possible amélioration :
+ Condisération des données de la températures dans le code : `sp = loc.get_solarposition(run_bsrn_data.index, temperature=25)`
+ Condisération des données sur la turbidité dans le code : `cs = loc.get_clearsky(run_bsrn_data.index, model='ineichen', linke_turbidity=2)`
+ Le choix d'un meilleur modèle de ciel clair.



## Importation of satellite estimates data (SARAH-3)
Pour télécharger les données de SARAH-3, il faut tout d'abord se créer un compte sur le site de CMSAF : 
+ Step 1 : [REGISTRATION](https://wui.cmsaf.eu/safira/action/viewLogin;jsessionid=EC9BCA5ED19EE259B6281E188D13FA1D.ku_2)
+ Step 2 : [Choose data](https://wui.cmsaf.eu/safira/action/viewProduktSearch)
+ Sélection les informations utiles dans la partie *ADVANCED SEARCH* :
  - `Product group` : Climate Data Records
  - `Product family` : SARAH ed. 3.0
  - `Product name ` : SIS - Surface incoming shortwave radiation ou DNI - Direct Normalised Irradiance (en fonction du besoin)
  - `Temporal resolution` : Instantaneous ou Daily ou Monthly (en fonction du besoin)
  - Cliquer sur `Show`
+ Cliquer sur le liens des données dans la colonne : `Name / Details`
+ Cliquer sur `Change projection / spatial resolution / domain` dans la partie *PRODUCT ADAPTATIONS*
+ Choisir les coordonnées géographique de la zone avec les angles de latitude et longitude pour  sélectionner uniquement les données utiles pour l'étude, cliquer sur `PREVIEWS` pour vérifier que la zone est correcte.
+ Cliquer sur `Proceed to time range selection`
+ Sur la page de *Specification of time range* choisir le début et la fin du jeux de données:
  - `Begin`: 2008-12-01
  - `End`: 2024-04-01
+ Cliquer sur  `Add to order cart`
+ Cliquer sur  `Place an order`
+ Suivre les instructions reçu par mail (2 jours pour les données de 2008-12-01 à 2024-04-01)



## Adapation of SARAH-3 data

Les données de SARAH-3 sont relativement éloigné des données des stations du réseau  IOS-net, ainsi une adaptation des données satellite est nécessaire afin que les données satellite soient plus proche des mesures au sol. Cette adapation doit être effectué sur le GHI et le DNI indépendamment.

![image](https://github.com/user-attachments/assets/2ea7e3f3-df0e-403d-96fe-8141f7a8075a)


## Classification SARAH-3 data



# Source
[1] : https://galilee.univ-reunion.fr/thredds/catalog.html
