# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:56:49 2024

@author: erwan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import datamanip

import time
import pvlib
from pvlib.location import Location
from pandas.plotting import register_matplotlib_converters



'bizarre'
#df_oldbsrn_irrad = pd.read_csv('data_brute/reunion/oldbsrn_irrad.csv', sep=';', index_col=0)
#df_oldbsrnsec_irrad = pd.read_csv('data_brute/reunion/oldbsrnsec_irrad.csv', sep=';', index_col=0)




#-------------------------------------------------------
#      Import data  
#-------------------------------------------------------

df_reunion_plaineparcnational = pd.read_csv('data_brute/reunion/plaineparcnational_irrad.csv', sep=';', index_col=0) #raw data
df_test = df_reunion_plaineparcnational['2020':'2021'] # selective data
data_geo = pd.read_csv('data_geo_all_station.csv', sep=';', index_col=0) # geographic data
df_one_columns = datamanip.one_column_ghi_dhi(df_test) # simplify data
df_estim_dni = datamanip.estimation_dni(df_one_columns, data_geo, 'plaineparcnational', 'Indian/Reunion') # data with DNI estimation
df_filter = datamanip.quality_of_bsrn(df_estim_dni) # useful data


#-------------------------------------------------------
#     Show estimate data of DNI in-situ  
#-------------------------------------------------------

datamanip.affiche_courbe_dhi_dni_ghi('plaineparnational', df_filter)


