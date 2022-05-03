# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:11:41 2022

@author: jocke
"""
import pandas as pd
import numpy as np
import numpy_financial as npf
import functions as fun
import time
import fa_new



def fitness_func_NPV(solution):
    """
    Returns the NPV value (used as fitness value in GA)
    """

    ESS_capacity, ESS_power = solution[0], solution[1]
    
    cashflow_each_year = [-((ESS_capacity_cost*ESS_capacity) + (ESS_power*ESS_power_cost))] #First Year is just capex cost and negative as it is a cost
    ESS_capacity_year = 0 #Starts with zero energy in the storage
    for year in range(1,11): # starts at year 1 and includes year 10 as the lifetime of ESS is 10 years (battery) 
        
        Schedule = fun.ESS_schedule(ESS_capacity_size=ESS_capacity, ESS_power=ESS_power,
                                        Energy_hourly_cost=Energy_hourly_cost,
                                        Average_median_cost_day=Average_median_cost_day,
                                        Energy_hourly_use=Energy_hourly_use,
                                        ESS_discharge_eff=ESS_discharge_eff, ESS_charge_eff=ESS_charge_eff, Year = year, ESS_capacity_prev_year= ESS_capacity_year)
    
                
        #This calculates the cost of buying and using the ESS storage, as well as the profits of sell energy from it, and inputs that into an array for each year.
        #This does not include the energy used by the user. (Aka the load demand), but the schedule is designed from that schedule
        New_schedule = Schedule[0]
        ESS_capacity_year = Schedule[1]  #Inputs the preveious years ess capacity to next years
        cashflow_each_year.append(fun.cashflow_yearly_NPV(schedule_load = New_schedule[:, 0], schedule_discharge = New_schedule[:,1], demand_cost = Energy_hourly_cost,
                                                          Variable_O_and_M_cost = Variable_ESS_O_and_M_cost, Fixed_O_and_M_cost = Fixed_ESS_O_and_M_cost, ESS_power = ESS_power))

    fitness = fun.Fitness_NPV(discount_rate = Discount_rate, cashflows = cashflow_each_year)
    return -fitness #negative as the FF algo want to minimize the fitness function


def fitness_func_LCOS(solution):
    ESS_capacity, ESS_power = solution[0], solution[1]
    CAPEX = ((ESS_capacity_cost*ESS_capacity) + (ESS_power*ESS_power_cost)) 
    Cost_yearly = []
    Energy_yearly = []
    ESS_capacity_year = 0 #Starts with zero energy in the storage
    
    for year in range(1,11): # starts at year 1 and includes year 10 as the lifetime of ESS is 10 years (battery)
        
        Schedule = fun.ESS_schedule(ESS_capacity_size=ESS_capacity, ESS_power=ESS_power,
                                        Energy_hourly_cost=Energy_hourly_cost,
                                        Average_median_cost_day=Average_median_cost_day,
                                        Energy_hourly_use=Energy_hourly_use,
                                        ESS_discharge_eff=ESS_discharge_eff, ESS_charge_eff=ESS_charge_eff, Year = year, ESS_capacity_prev_year= ESS_capacity_year)
    
                
        #This calculates the cost of buying and using the ESS storage, as well as the profits of sell energy from it, and inputs that into an array for each year.
        #This does not include the energy used by the user. (Aka the load demand), but the schedule is designed from that schedule
        New_schedule = Schedule[0]
        ESS_capacity_year = Schedule[1]  #Inputs the preveious years ess capacity to next years
        Cost_yearly.append(fun.Cost_yearly_LCOS(schedule_load = New_schedule[:,0], schedule_discharge = New_schedule[:,1], demand_cost = Energy_hourly_cost,
                             Fixed_O_and_M_cost = Fixed_ESS_O_and_M_cost, Variable_O_and_M_cost = Variable_ESS_O_and_M_cost, ESS_power = ESS_power))
        
        Energy_yearly.append(np.sum(New_schedule[:,1]))

    fitness = fun.Fittnes_LCOS(discount_rate = Discount_rate, CAPEX = CAPEX, Yearly_cost = Cost_yearly, Yearly_energy_out = Energy_yearly)
    return fitness #returns LCOS in Euro/kWh

#-------------------------------------------------------end of functions------------------------


# ----------Gets the average cost for each day, and the hourly cost at each hour during the year--------

Electricity_price_read = np.genfromtxt(
    "sto-eur17.csv", delimiter=",")  # Prices in EUR/MWh
El_cost_year = []
El_cost_average_day = []

for i in range(365):
    for k in Electricity_price_read[i][0:24]:
        El_cost_year.append((k/1000)*1.11) #Prices in Euro/kWh, by dividing by 1000, times 1.1 to get 2022 euro values due to inflation
          
    El_cost_average_day.append(((Electricity_price_read[i][24])/1000)*1.1)  #Prices in Euro/kwh that is why we are dividing by 1000, times 1.1 to get 2022 values



# -------------Read load data for each hour of a year of house 59---------
El_data_read = pd.read_csv("home59_hall687_sensor1506c1508_electric-mains_electric-combined.csv",
                           header=None)

# Average W used for that hour (giving the Wh) from second of january 2017
El_data_59 = El_data_read[1][2038:8760+2038]

power_load_59 = []
for i in El_data_59:
    power_load_59.append((i/1000))  # diveide by 1000 to get it in kWh, times 5 for 5 houses

# --------------------------------------------------------------------------


# ---------------Input values (non changable)---------------------------------------'
# ------For Schedule inputs-----------

Energy_hourly_cost = np.array(El_cost_year)     #Prices in Euro/kWh
Average_median_cost_day = np.array(El_cost_average_day) 
Energy_hourly_use = np.array(power_load_59)
ESS_charge_eff = 0.9
ESS_discharge_eff = 0.9


# Important to note that the maximum SoC for the battery is calculated in the schdule function
# Only import is the Total max size that is also used for calculating the cost


# ------For NPV/LCOE inputs -------------
Lifetime_battery = 10  # in years
ESS_capacity_cost = 427.31   # in Euro(2022) per kWh (CAPEX) all cost included
ESS_power_cost = 1710.2  # in Euro(2022) per kW (all cost included)
Fixed_ESS_O_and_M_cost = 4.19  # in Euro(2022) per kWh-year
Variable_ESS_O_and_M_cost = 0.488/1000 # in Euro(2022) per kWh-year 
Discount_rate = 0.08 #8 percent


#------------Setup of parameters for FF algorithm------------

#the FF algo want to minimize the fitness function!

n = 10 #number of agents (fireflies) Comparable to number of solution in GA
fitness_function = fitness_func_NPV       #fitness function to be used
lb = 1  #lower bound of search space (plot axis)
ub = 2000 #Higher bound of search space (plot axis)
dimensions = 2 #search space dimension (for us 2 one for ESS capcity and one for ESS power)
iteration = 5  #number of iterations the algorithm will run


csi = 1     #mutal attraction value
psi =  1 #Light absoprtion coefficent
alpha0 = 1   #initial value of the free randomization parameter alpha what alpha starts on iteration 1
alpha1 = 0.1 #final value of the free randomization parameter alpha what alpha is going to for a value exponentionally depening on iteration t
norm0 = 0   #first parameter for a normal (Gaussian) distribution 
norm1 = 0.1  #second parameter for a normal (Gaussian) distribution #as we are looking at ints these are not normal gassuian

#-----------set up FF algorithm----------
start = time.time() 
alh = fa_new.fa(n = n, function = fitness_function, lb = lb, ub = ub, dimension = dimensions,
                         iteration = iteration, csi = csi, psi =  psi, alpha0 = alpha0,
                        alpha1 = alpha1, norm0 = norm0, norm1 = norm1)

end = time.time()
print("here are the agents, with best fittness", alh.get_Gbest())
if fitness_function == fitness_func_LCOS:
    print("fitness value of bes solution: ", fitness_function(alh.get_Gbest()))
elif fitness_function == fitness_func_NPV:
    print("fitness value of bes solution: ", -fitness_function(alh.get_Gbest())) #negative value as we return the fitness value for NPV as a negative as the algo want to minimize it, then we have to flip the sign as we then have found the maximum
print("Time to run optimization: ", abs(start-end))