import os
import glob
import pvlib
from pvlib.location import Location
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
    df_minute_mean = df_1.resample('T').mean()

    # Returns the DataFrame with outliers replaced by np.nan
    return df_minute_mean

#############################################################################################################
def check_ghi_limits(data):
    """
    Vérifie si les valeurs de la colonne 'ghi' sont inférieures ou égales aux valeurs de la colonne 'physical_limit_ghi'.
    Considère les NaN dans 'ghi' comme satisfaisant la condition.
    
    Parameters:
    df (pd.DataFrame): Le DataFrame contenant les colonnes 'ghi' et 'physical_limit_ghi'.
    
    Returns:
        str: Message indiquant si toutes les valeurs de 'ghi' sont inférieures ou égales à 'physical_limit_ghi' ou sont NaN.
    """
    df = data.copy()
    # Vérifie si les colonnes nécessaires sont présentes dans le DataFrame
    if 'ghi' not in df.columns or 'physical_limit_ghi' not in df.columns:
        raise ValueError("Le DataFrame doit contenir les colonnes 'ghi' et 'physical_limit_ghi'")
    
    # Crée une nouvelle colonne 'ghi_within_limit'
    df['ghi_within_limit'] = (df['ghi'] <= df['physical_limit_ghi']) | df['ghi'].isna()
    
    # Vérifie si toutes les valeurs de 'ghi_within_limit' sont True
    if df['ghi_within_limit'].all():
        return True
    else:
        return False
        
#############################################################################################################       
def check_dhi_limits(data):
    """
    Vérifie si les valeurs de la colonne 'ghi' sont inférieures ou égales aux valeurs de la colonne 'physical_limit_ghi'.
    Considère les NaN dans 'ghi' comme satisfaisant la condition.
    
    Parameters:
    df (pd.DataFrame): Le DataFrame contenant les colonnes 'ghi' et 'physical_limit_ghi'.
    
    Returns:
        str: Message indiquant si toutes les valeurs de 'ghi' sont inférieures ou égales à 'physical_limit_ghi' ou sont NaN.
    """
    df = data.copy()
    # Vérifie si les colonnes nécessaires sont présentes dans le DataFrame
    if 'dhi' not in df.columns or 'physical_limit_dhi' not in df.columns:
        raise ValueError("Le DataFrame doit contenir les colonnes 'dhi' et 'physical_limit_dhi'")
    
    # Crée une nouvelle colonne 'ghi_within_limit'
    df['dhi_within_limit'] = (df['dhi'] <= df['physical_limit_dhi']) | df['dhi'].isna()
    
    # Vérifie si toutes les valeurs de 'ghi_within_limit' sont True
    if df['dhi_within_limit'].all():
        return True
    else:
        return False
        
#############################################################################################################
def check_dni_limits(data):
    """
    Vérifie si les valeurs de la colonne 'ghi' sont inférieures ou égales aux valeurs de la colonne 'physical_limit_ghi'.
    Considère les NaN dans 'ghi' comme satisfaisant la condition.
    
    Parameters:
    df (pd.DataFrame): Le DataFrame contenant les colonnes 'ghi' et 'physical_limit_ghi'.
    
    Returns:
        str: Message indiquant si toutes les valeurs de 'ghi' sont inférieures ou égales à 'physical_limit_ghi' ou sont NaN.
    """
    df = data.copy()
    # Vérifie si les colonnes nécessaires sont présentes dans le DataFrame
    if 'dni' not in df.columns or 'physical_limit_dni' not in df.columns:
        raise ValueError("Le DataFrame doit contenir les colonnes 'dni' et 'physical_limit_dni'")
    
    # Crée une nouvelle colonne 'ghi_within_limit'
    df['dni_within_limit'] = (df['dni'] <= df['physical_limit_dni']) | df['dni'].isna()
    
    # Vérifie si toutes les valeurs de 'ghi_within_limit' sont True
    if df['dni_within_limit'].all():
        return True
    else:
        return False

#############################################################################################################
def save_dni_estimation(data_geo):
    """
    Save DNI estimation in .csv file
    
    Args:
        data_geo (dataframe) : dataframe contains geographical data of stations
    Return:
        none

    Example : 
        data_geo = pd.read_csv('data_geo_adaptation.csv', sep=';', index_col=0)
    """
    for name in data_geo.index.tolist():
        try:
            df = pd.read_csv(f'data_raw_UTC_minute/{name}_irrad.csv', sep=';', index_col=0)
            good_df = one_column_ghi_dhi(df)
            dni_estimation = estimation_dni_physical_limits(good_df, data_geo,name, 'Indian/Reunion')
            qc_data = quality_of_bsrn(dni_estimation)
            if check_ghi_limits(qc_data) is True and check_dhi_limits(qc_data) is True and check_dni_limits(qc_data) is True:
                df_round = qc_data.round(2)
                df_round.to_csv(f'data_UTC_valid_minute/{name}.csv', sep=';', index=True)
                print(name + ": Backup Success")
        except Exception as e:
            print(name + f": Error: {e}")

