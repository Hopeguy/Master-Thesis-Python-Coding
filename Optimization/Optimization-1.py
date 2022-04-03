# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:39:23 2022

@author: jocke
"""

import pandas as pd
import numpy as np
import time

### Function for solving the ESS schedule ###

def ESS_schedule(ESS_capacity_max, ESS_power, Energy_hourly_cost, Average_median_cost_day, Energy_hourly_use):
    """
    Where:
    ESS_capacity_max is in kWh;
    ESS_power in kW;
    Energy_hourly cost in pence and all hours of a year (list of 8760 hours);
    Average_median_cost_day in pence for each day of a year (list if 365 days);
    Energy_hourly_use in kWh in load demand from user for each hour in a year (list of 8760 hours)
    
    """
    schedule_capacity = np.zeros((8760,2)) #Matrix to store Capacity and power input/output for each hour.
    ESS_capacity = 0 #Starts at zero energy in the ESS unit
    hour_year = 0
    for day_averge_cost in Average_median_cost_day:
        
        for hour_day in range(1,25):
            
            if day_averge_cost > Energy_hourly_cost[hour_year]: # Checks if the average cost is higher than current (hourly) (We want to charge ESS)
                if ESS_capacity < ESS_capacity_max:             # Checks if the ESS capaicty is full or not
                    if ESS_capacity + ESS_power < ESS_capacity_max: #Checks if we can charge the battery with maximum Power 
                        ESS_capacity += ESS_power   #charges the ESS with its maximum power
                        schedule_capacity[hour_year, 0] = ESS_power
                        schedule_capacity[hour_year, 1] = ESS_capacity  #gives an list with how charged the ESS for each hour
                    else:
                        schedule_capacity[hour_year, 0] = ESS_capacity_max - ESS_capacity
                        ESS_capacity = ESS_capacity_max
                        schedule_capacity[hour_year, 1] = ESS_capacity
                else:
                    schedule_capacity[hour_year, 1] = ESS_capacity #If the capacity is full, nothing happends and here we just include it to get a list of its behavior
                        
                        
            if day_averge_cost < Energy_hourly_cost[hour_year]:# Checks if the average cost is lower than current (hourly) (We want to discharge ESS)
                   if ESS_capacity > 0:             # Checks if the ESS capaicty is empty or not
                       if Energy_hourly_use[hour_year] > ESS_power:  #Checks here if the energy used by the consumer is above the maximum power that can be drawn from the ESS
                           if ESS_capacity > ESS_power: #Checks if we can charge the battery with maximum Power 
                               ESS_capacity -= ESS_power   #charges the ESS with its maximum power
                               schedule_capacity[hour_year, 0] = -ESS_power #Sets the schedule to a negative value as it uses energy from the ESS
                               schedule_capacity[hour_year, 1] = ESS_capacity
                           else:
                               schedule_capacity[hour_year, 0] = -ESS_capacity #This case is when we have less than maximum power, and then uses up the last energy availbale in the ESS
                               ESS_capacity = 0
                               schedule_capacity[hour_year, 1] = ESS_capacity
                       if Energy_hourly_use[hour_year] < ESS_power:  #When the powered used by consumer is less then the maximum power by the ESS, we here then use the maximu we can but that is less than ESS power max
                           if ESS_capacity > Energy_hourly_use[hour_year]:      #Checks that the ESS have above the energy we want to discharge from it
                               schedule_capacity[hour_year, 0] = Energy_hourly_use[hour_year]
                               ESS_capacity -= Energy_hourly_use[hour_year]
                               schedule_capacity[hour_year, 1] = ESS_capacity
                           elif ESS_capacity < Energy_hourly_use[hour_year]: #This is the case when we dont enough energy to discharge from the ESS so we take all we can take from this hour.
                               schedule_capacity[hour_year, 0] -= ESS_capacity
                               ESS_capacity = 0
                               schedule_capacity[hour_year, 1] = ESS_capacity
                        
            
            hour_year += 1  #At what hour we are in during the year
                          

    return schedule_capacity
    
######-------------------------------------------------------


### Fittness function -------------------

def Fittnes():
    solution = 0
    
    
    return solution


El_data_read = pd.read_csv("home59_hall687_sensor1506c1508_electric-mains_electric-combined.csv",
                      header = None)
Electricity_price_read = pd.read_csv("Electricity_prices_uk_nordpool_2022_03_01.csv", header= 0)

El_data_59 = El_data_read[1][2038:8760+2038] #Average W used for that hour (giving the Wh) from second of january

power_load_59 = []
for i in El_data_59:
    power_load_59.append(i/1000)   #in kWh

    
El_power_cost_hourly = Electricity_price_read['GBP/MWh']/10 #In pence per kWh, right now from 1 march 2022

El_cost_day = []
for i in El_power_cost_hourly:
    El_cost_day.append(i)
#to get a year of electricity cost data we copy this 365 times




## Gets the average cost for each day, and the hourly cost at each hour during the year


## Base way to do it:   

El_cost_year = []
El_cost_average_day = []
for i in range(365):
    total_day_cost = 0
    for k in El_cost_day:
        total_day_cost += k
        El_cost_year.append(k)
    
    El_cost_average_day.append(total_day_cost/24)
    total_day_cost = 0

##----------------------------------

### Calculate Base case cost ####
Base_case_cost = 0 #  in  pence

for Count, El in enumerate(El_data_59):
    Base_case_cost += (El/1000)*(El_cost_year[Count]) #Divide by 1000 to get how many kWh
    


### Input values for solution matrix ###

Battery_size = list(range(1, 101)) #kWh
Battery_power = list(range(1, 101)) #kW

start = time.time()

Schedule = ESS_schedule(ESS_capacity_max = 1, ESS_power = 0.1, Energy_hourly_cost = El_cost_year, Average_median_cost_day = El_cost_average_day, Energy_hourly_use = power_load_59)

end = time.time()
print(end - start)
