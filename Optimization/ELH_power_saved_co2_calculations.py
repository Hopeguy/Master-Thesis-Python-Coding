
import numpy as np
import pandas as pd


def Electricity_heater_load(Power, Heating_load):
    """"
    Power in kW
    Heating load in kWh (An array with 8760 hours)
    Efficency set to 95% (Source needed)
    Output is the new lower heating load that has been taken care of with the electrical heater (array of 8760 hours)
    Electrical load from the electrical heater (array of 8760 hours)
    """

    global New_heating_load

    # Depending on source but assumed to be 95% (Source  "UKSupplyCurve")
    Efficency = 0.95
    Electricity_load = np.zeros(8760)
    New_heating_load = np.zeros(8760)
    # goes through the array with the load demand
    for count, load in enumerate(Heating_load):
        if load < Power:  # if the load is less then the power of the electrical heater
            # Electricity load is increase by the load divided by the efficency
            Electricity_load[count] = load/Efficency
            # as the load was lower then the power zero heating load is left this hour
            New_heating_load[count] = 0
        elif load > Power:
            # When the load is higher than the Power of the electrical heater, the new electricty this hour is the power divided by the efficency
            Electricity_load[count] = Power/Efficency
            # The heat load this hour is the load minus the power that was removed by the electrical heater
            New_heating_load[count] = load - Power

    return Electricity_load, New_heating_load



# --------------------Read load data for each hour both heating and electrical load---------
Load_data_read = pd.read_csv("Load_data_electricit_heating_2017.csv", header=0) #Takes values from January
Heating_load_pd = Load_data_read["Heating demand [kW]"]

Heating_load = np.zeros(8760) #in kWh
Electricity_load = np.zeros(8760) #in kWh

for count, i in enumerate(Heating_load_pd):
    Heating_load[count] = i


Heating_hourly_use = np.array(Heating_load)

CO2_coeff_emi = 0.442 #Kg-co2-eq/kWh
gas_heater_Eff = 0.95

sum_heating = np.sum(Heating_hourly_use)/gas_heater_Eff
Co2_emissions = sum_heating*CO2_coeff_emi/1000 #in tons
print(sum_heating, Co2_emissions)

#it goes GA: NPV, LCOS, FF: NPV, LCOS for 200 iterations case 3
ELH_array = [1528.21529240573, 898.324884827645, 1018.591381, 0.1]
New_heating_Array = [] #How much heating is left to do by the gas one i kWh

for i in ELH_array:

    New_heating = Electricity_heater_load(i, Heating_hourly_use)
    New_heating_Array.append(np.sum(New_heating[1]))


print(New_heating_Array)

for count, i in enumerate(New_heating_Array):
    New_heating_Array[count] = abs(i-sum_heating)/1000 #Saved MWh heated with gas


print("Heating now by ELH instead of gas heater: ", New_heating_Array)
print("procentual compared to before installment: ", New_heating_Array/(sum_heating/1000))

saved_co = []

for i in New_heating_Array:
    saved_co.append(i*CO2_coeff_emi) #in tons

print("Saved CO_2 in tons: ", saved_co)
print("Saved co2 procent :" , saved_co/Co2_emissions)