
from ast import Load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


Load_data_read = pd.read_csv("Load_data_electricit_heating_2017.csv", header=0) #Takes values from January
Electricity_load_pd = Load_data_read["Electricty [kW]"]
Heating_load_pd = Load_data_read["Heating demand [kW]"]

Heating_load = np.zeros(8760)
Electricity_load = np.zeros(8760)

for count, i in enumerate(Heating_load_pd):
    Heating_load[count] = i

for count, i in enumerate(Electricity_load_pd):
    Electricity_load[count] = i



def Electricity_heater_load(Power, Heating_load):
    """"
    Power in kW
    Heating load in kWh (An array with 8760 hours)
    Efficency set to 95% (Source)
    Output is the new lower heating load that has been taken care of with the electrical heater (array of 8760 hours)
    Electrical load from the electrical heater (array of 8760 hours)
    """
    
    Efficency = 0.95 # Depending on source but assumed to be 95 (Source  "UKSupplyCurve")
    Electricity_load_heater = np.zeros(8760)
    New_heating_load = np.zeros(8760)
    for count, load in enumerate(Heating_load): #goes through the array with the load demand
        if load < Power:                        #if the load is less then the power of the electrical heater
            Electricity_load_heater[count] = load/Efficency    #Electricity load is increase by the load divided by the efficency
            New_heating_load[count] = 0                 #as the load was lower then the power zero heating load is left this hour
        elif load > Power:
            Electricity_load_heater[count] = Power/Efficency   # When the load is higher than the Power of the electrical heater, the new electricty this hour is the power divided by the efficency
            New_heating_load[count] = load - Power #The heat load this hour is the load minus the power that was removed by the electrical heater

    return Electricity_load_heater, New_heating_load
    


Power = 5

Electricity_load_heater, New_heating_load  = Electricity_heater_load(Power = Power, Heating_load = Heating_load)


print(Electricity_load_heater)
print(New_heating_load)
print(Heating_load)

