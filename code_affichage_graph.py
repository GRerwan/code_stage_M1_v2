# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:56:49 2024

@author: erwan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import datamanip



'''
#____south africa_______________
df_durban_irrad = pd.read_csv('data_brute/south_africa/durban_irrad.csv', sep=';', index_col=0)


#____seychelles_______________
df_amitie_irrad = pd.read_csv('data_brute/seychelles/amitie_irrad.csv', sep=';', index_col=0)
df_anseboileau_irrad = pd.read_csv('data_brute/seychelles/amitie_irrad.csv', sep=';', index_col=0)


#____mauritius_______________
df_mrtbrasdeau_irrad = pd.read_csv('data_brute/mauritius/mrtbrasdeau_irrad.csv', sep=';', index_col=0)
df_reservetortues_irrad = pd.read_csv('data_brute/mauritius/reservetortues_irrad.csv', sep=';', index_col=0)
df_uomreduit_irrad = pd.read_csv('data_brute/mauritius/uomreduit_irrad.csv', sep=';', index_col=0)
df_vacoas_irrad = pd.read_csv('data_brute/mauritius/vacoas_irrad.csv', sep=';', index_col=0)

#____madagascar_______________
df_antananarivo_irrad = pd.read_csv('data_brute/mada/antananarivo_irrad.csv', sep=';', index_col=0)
df_diego_irrad = pd.read_csv('data_brute/mada/diego_irrad.csv', sep=';', index_col=0)

#____reunion_______________
df_aurere_irrad = pd.read_csv('data_brute/reunion/aurere_irrad.csv', sep=';', index_col=0)
df_braspanonmoreau_irrad = pd.read_csv('data_brute/reunion/braspanonmoreau_irrad.csv', sep=';', index_col=0)
df_cavernedufour_irrad = pd.read_csv('data_brute/reunion/cavernedufour_irrad.csv', sep=';', index_col=0)
df_cilaosbrassec_irrad = pd.read_csv('data_brute/reunion/cilaosbrassec_irrad.csv', sep=';', index_col=0)
df_cilaospiscine_irrad = pd.read_csv('data_brute/reunion/cilaospiscine_irrad.csv', sep=';', index_col=0)
df_cilaosthermes_irrad = pd.read_csv('data_brute/reunion/cilaosthermes_irrad.csv', sep=';', index_col=0)
df_craterebory_irrad = pd.read_csv('data_brute/reunion/craterebory_irrad.csv', sep=';', index_col=0)
df_edfboisdenefles_irrad = pd.read_csv('data_brute/reunion/edfboisdenefles_irrad.csv', sep=';', index_col=0)
df_edflapossession_irrad = pd.read_csv('data_brute/reunion/edflapossession_irrad.csv', sep=';', index_col=0)
df_edfsaintandre_irrad = pd.read_csv('data_brute/reunion/edfsaintandre_irrad.csv', sep=';', index_col=0)
df_edfsaintleu_irrad = pd.read_csv('data_brute/reunion/edfsaintleu_irrad.csv', sep=';', index_col=0)
df_edfsaintpierre_irrad = pd.read_csv('data_brute/reunion/edfsaintpierre_irrad.csv', sep=';', index_col=0)
df_leportbarbusse_irrad = pd.read_csv('data_brute/reunion/leportbarbusse_irrad.csv', sep=';', index_col=0)
df_leportmairie_irrad = pd.read_csv('data_brute/reunion/leportmairie_irrad.csv', sep=';', index_col=0)
df_marla_irrad = pd.read_csv('data_brute/reunion/marla_irrad.csv', sep=';', index_col=0)
df_observatoirevolcan_irrad = pd.read_csv('data_brute/reunion/observatoirevolcan_irrad.csv', sep=';', index_col=0)
df_oldbsrn_irrad = pd.read_csv('data_brute/reunion/oldbsrn_irrad.csv', sep=';', index_col=0)
df_oldbsrnsec_irrad = pd.read_csv('data_brute/reunion/oldbsrnsec_irrad.csv', sep=';', index_col=0)
df_pitondesneiges_irrad = pd.read_csv('data_brute/reunion/pitondesneiges_irrad.csv', sep=';', index_col=0)
df_plaineparcnational_irrad = pd.read_csv('data_brute/reunion/plaineparcnational_irrad.csv', sep=';', index_col=0)
df_sainterosemairie_irrad = pd.read_csv('data_brute/reunion/sainterosemairie_irrad.csv', sep=';', index_col=0)
df_saintjosephmairie_irrad = pd.read_csv('data_brute/reunion/saintjosephmairie_irrad.csv', sep=';', index_col=0)
df_saintlouisjeanjoly_irrad = pd.read_csv('data_brute/reunion/saintlouisjeanjoly_irrad.csv', sep=';', index_col=0)
df_saintpaulcarat_irrad = pd.read_csv('data_brute/reunion/saintpaulcarat_irrad.csv', sep=';', index_col=0)
df_urbsrn_irrad = pd.read_csv('data_brute/reunion/urbsrn_irrad.csv', sep=';', index_col=0)
df_urbsrnsec_irrad = pd.read_csv('data_brute/reunion/urbsrnsec_irrad.csv', sep=';', index_col=0)
df_urmoufia_irrad = pd.read_csv('data_brute/reunion/urmoufia_irrad.csv', sep=';', index_col=0)
df_urtampon_irrad = pd.read_csv('data_brute/reunion/urtampon_irrad.csv', sep=';', index_col=0)


#____comores_______________
df_hahaya_irrad = pd.read_csv('data_brute/comores/hahaya_irrad.csv', sep=';', index_col=0)
df_ouani_irrad = pd.read_csv('data_brute/comores/ouani_irrad.csv', sep=';', index_col=0)



df_cilaosbrassec_irrad = pd.read_csv('data_brute/reunion/cilaosbrassec_irrad.csv', sep=';', index_col=0)
df_cilaospiscine_irrad = pd.read_csv('data_brute/reunion/cilaospiscine_irrad.csv', sep=';', index_col=0)
df_cilaosthermes_irrad = pd.read_csv('data_brute/reunion/cilaosthermes_irrad.csv', sep=';', index_col=0)
df_craterebory_irrad = pd.read_csv('data_brute/reunion/craterebory_irrad.csv', sep=';', index_col=0)
df_edfboisdenefles_irrad = pd.read_csv('data_brute/reunion/edfboisdenefles_irrad.csv', sep=';', index_col=0)
df_edflapossession_irrad = pd.read_csv('data_brute/reunion/edflapossession_irrad.csv', sep=';', index_col=0)
df_edfsaintandre_irrad = pd.read_csv('data_brute/reunion/edfsaintandre_irrad.csv', sep=';', index_col=0)
df_edfsaintleu_irrad = pd.read_csv('data_brute/reunion/edfsaintleu_irrad.csv', sep=';', index_col=0)
'''
#df_edfsaintpierre_irrad = pd.read_csv('data_brute/reunion/edfsaintpierre_irrad.csv', sep=';', index_col=0)
#df_leportbarbusse_irrad = pd.read_csv('data_brute/reunion/leportbarbusse_irrad.csv', sep=';', index_col=0)

'trop long pour vendredi'
#df_leportmairie_irrad = pd.read_csv('data_brute/reunion/leportmairie_irrad.csv', sep=';', index_col=0)

#df_marla_irrad = pd.read_csv('data_brute/reunion/marla_irrad.csv', sep=';', index_col=0)
#df_observatoirevolcan_irrad = pd.read_csv('data_brute/reunion/observatoirevolcan_irrad.csv', sep=';', index_col=0)

'bizarre'
#df_oldbsrn_irrad = pd.read_csv('data_brute/reunion/oldbsrn_irrad.csv', sep=';', index_col=0)
#df_oldbsrnsec_irrad = pd.read_csv('data_brute/reunion/oldbsrnsec_irrad.csv', sep=';', index_col=0)


#df_pitondesneiges_irrad = pd.read_csv('data_brute/reunion/pitondesneiges_irrad.csv', sep=';', index_col=0)
#df_plaineparcnational_irrad = pd.read_csv('data_brute/reunion/plaineparcnational_irrad.csv', sep=';', index_col=0)
#df_sainterosemairie_irrad = pd.read_csv('data_brute/reunion/sainterosemairie_irrad.csv', sep=';', index_col=0)
#df_saintjosephmairie_irrad = pd.read_csv('data_brute/reunion/saintjosephmairie_irrad.csv', sep=';', index_col=0)
#df_saintlouisjeanjoly_irrad = pd.read_csv('data_brute/reunion/saintlouisjeanjoly_irrad.csv', sep=';', index_col=0)
#df_saintpaulcarat_irrad = pd.read_csv('data_brute/reunion/saintpaulcarat_irrad.csv', sep=';', index_col=0)
#df_urbsrn_irrad = pd.read_csv('data_brute/reunion/urbsrn_irrad.csv', sep=';', index_col=0)
#df_urbsrnsec_irrad = pd.read_csv('data_brute/reunion/urbsrnsec_irrad.csv', sep=';', index_col=0)
df_urmoufia_irrad = pd.read_csv('data_brute/reunion/urmoufia_irrad.csv', sep=';', index_col=0)
#df_urtampon_irrad = pd.read_csv('data_brute/reunion/urtampon_irrad.csv', sep=';', index_col=0)

#____comores_______________
#df_hahaya_irrad = pd.read_csv('data_brute/comores/hahaya_irrad.csv', sep=';', index_col=0)
#df_ouani_irrad = pd.read_csv('data_brute/comores/ouani_irrad.csv', sep=';', index_col=0)















