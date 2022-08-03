import numpy as np
import numpy_financial as npf
import pandas as pd
import matplotlib.pyplot as plt
"""ELH_power = 30 #kW
ELH_power_cost = 331.2 #Euro/kW
ELH_cost = ELH_power*ELH_power
lifetime = 15 #years
interest_rate = 0.08 # 8%


pmt = npf.pmt(pv = ELH_cost, nper= lifetime, rate=interest_rate) #Monthly over the 15 years

print(pmt) 

Residual_value = pmt*5
print(Residual_value) #Euro
"""


Load_data_read = pd.read_csv("Load_data_electricit_heating_2017.csv", header=0) #Takes values from January, Empty data in 7976 set to 0
Load_temp_read = pd.read_csv("Temperature.csv", header=0)
temperature_load_pd = Load_temp_read["Temp"]
Electricity_load_pd = Load_data_read["Electricty [kW]"]
Heating_load_pd = Load_data_read["Heating demand [kW]"]

Heating_load = np.zeros(8760)
Electricity_load = np.zeros(8760)
day_average_heating = np.zeros(365)
day_average_temp = np.zeros(365)

for count, i in enumerate(Heating_load_pd):
    Heating_load[count] = i

for count, i in enumerate(Electricity_load_pd):
    Electricity_load[count] = i


for k in range(365):
    for i in range(24):
        day_average_heating[k] = np.sum(Heating_load[k*24:(k*24)+24])/24
        day_average_temp[k] = np.sum(temperature_load_pd[k*24:(k*24)+24])/24


plt.plot(day_average_heating)

