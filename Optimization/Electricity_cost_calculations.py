

import pandas as pd
import numpy as np
import numpy_financial as npf
import functions as fun
import time
import pygad
import matplotlib.pyplot as pl



Electricity_price_read = np.genfromtxt(
    "sto-eur17.csv", delimiter=",")  # Prices in EUR/MWh
El_cost_year = []
El_cost_average_day = []

for i in range(365):
    for k in Electricity_price_read[i][0:24]:
        El_cost_year.append((k/1000)*1.1) #Prices in Euro/kWh, by dividing by 1000, times 1.1 to get 2022 euro values
          
    El_cost_average_day.append(((Electricity_price_read[i][24])/1000)*1.1)  #Prices in Euro/kWh that is why we are dividing by 1000, times 1.1 to get 2022 values


# --------------------Read load data for both Electricity and Heating--------
Load_data_read = pd.read_csv("Load_data_electricit_heating_2017.csv", header=0) #Takes values from January, Empty data in 7976 set to 0
Electricity_load_pd = Load_data_read["Electricty [kW]"]

Electricity_load = np.zeros(8760)

for count, i in enumerate(Electricity_load_pd):
    Electricity_load[count] = i


total_electricy_need = np.sum(Electricity_load) #in kWh 
Total_cost = np.sum(Electricity_load*El_cost_year) #in Euro
