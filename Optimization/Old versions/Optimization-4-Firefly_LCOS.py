# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:11:41 2022

@author: jocke
"""
from multiprocessing.dummy import Array
import pandas as pd
import numpy as np
import numpy_financial as npf
import functions as fun
import time
import fa_new_version_4
import matplotlib.pyplot as plt


def fitness_func_LCOS(solution):
    ESS_capacity, ESS_power = solution[0], solution[1]
    CAPEX = ((ESS_capacity_cost*ESS_capacity) + (ESS_power*ESS_power_cost)) 
    Cost_yearly = []
    Energy_yearly = []
    ESS_capacity_year = 0 #Starts with zero energy in the storage
    
    for year in range(10): # starts at year 1 and includes year 10 as the lifetime of ESS is 10 years (battery)
        
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


# --------------------Read load data for both Electricity and Heating--------
Load_data_read = pd.read_csv("Load_data_electricit_heating_2017.csv", header=0) #Takes values from January, Empty data in 7976 set to 0
Electricity_load_pd = Load_data_read["Electricty [kW]"]

Electricity_load = np.zeros(8760)

for count, i in enumerate(Electricity_load_pd):
    Electricity_load[count] = i


# ---------------Input values (non changable)---------------------------------------'
# ------For Schedule inputs-----------

Energy_hourly_cost = np.array(El_cost_year)     #Prices in Euro/kWh
Average_median_cost_day = np.array(El_cost_average_day) 
Energy_hourly_use = np.array(Electricity_load)
ESS_charge_eff = 0.9
ESS_discharge_eff = 0.9


# Important to note that the maximum SoC for the battery is calculated in the schdule function
# Only import is the Total max size that is also used for calculating the cost


# ------For NPV/LCOE inputs -------------
Lifetime_battery = 10  # in years
ESS_capacity_cost = 427.31   # in Euro(2022) per kWh (CAPEX) all cost included
ESS_power_cost = 1710.2  # in Euro(2022) per kW (all cost included)
Fixed_ESS_O_and_M_cost = 4.19  # in Euro(2022) per kW-year
Variable_ESS_O_and_M_cost = 0.488/1000 # in Euro(2022) per kWh-year 
Discount_rate = 0.08 #8 percent
Peak_cost = 5.92/1.1218 #5.92 dollar (2022) per kW (max per month) change to euro: 1 euro is 1.1218 USD january 1 2022

#------------Setup of parameters for FF algorithm------------

#the FF algo want to minimize the fitness function!

<<<<<<< HEAD:Optimization/Optimization-4-Firefly.py
n = 20 #number of agents (fireflies) Comparable to number of solution 
fitness_function = fitness_func_NPV       #fitness function to be used
lb1, ub1= 0.1, 2000 
=======
n = 10 #number of agents (fireflies) Comparable to number of solution 
fitness_function = fitness_func_LCOS       #fitness function to be used
lb1, ub1= 0.1, 8000 
>>>>>>> 7a73a6085762edc14232f5c688e297300a9c5229:Optimization/Old versions/Optimization-4-Firefly_LCOS.py
lb2, ub2 = 0.1, np.max(Energy_hourly_use)  #lower bound of search space (plot axis)
dimensions = 2 #search space dimension (for us 2 one for ESS capcity and one for ESS power)
iteration = 20  #number of iterations the algorithm will run


csi = 1  #mutal attraction value
psi =  1 #Light absoprtion coefficent
alpha0 = 1  #initial value of the free randomization parameter alpha what alpha starts on iteration 1
alpha1 = 0.1 #final value of the free randomization parameter alpha what alpha is going to for a value exponentionally depening on iteration t
norm0 = 0   #first parameter for a normal (Gaussian) distribution 
norm1 = 0.1  #second parameter for a normal (Gaussian) distribution #as we are looking at ints these are not normal gassuian


#-----------set up FF algorithm----------
Result_10_tries = [[],[],[]]

for i in range(10):

    start = time.time() 
    alh = fa_new_version_4.fa(n = n, function = fitness_function, lb1 = lb1, ub1 = ub1, lb2 = lb2, ub2 = ub2, dimension = dimensions,
                        iteration = iteration, csi = csi, psi =  psi, alpha0 = alpha0,
                        alpha1 = alpha1, norm0 = norm0, norm1 = norm1)
    end = time.time()

    
    Result_10_tries[0].append(alh.get_Gbest())
    Result_10_tries[1].append(-fitness_function(alh.get_Gbest()))
    Result_10_tries[2].append(abs(end-start))
    print(Result_10_tries[1])


cost_investment = -((ESS_capacity_cost*ESS_capacity) + (ESS_power*ESS_power_cost))

cashflow_divided[6] = cost_investment  #Adds the investment cost
cashflow_divided[7] = -fitness_function(alh.get_Gbest()) #Adds the NPV to the list of all the cost and profits
#Plot of all the cost and profits over the tne years divided
print("Profit kWh: ",cashflow_divided[0], "Profit peak: ", cashflow_divided[1])
print("cost charge: ", cashflow_divided[2], "cost OnM fixed: ", cashflow_divided[3], "cost OnM variable: ", cashflow_divided[4])
print("Total cashflow: ", cashflow_divided[5], "Cost investment: ", cost_investment, "NPV or LCOS: ", cashflow_divided[7])

fig, ax = plt.subplots()
ax.bar(["Profit_kWh", "Profit_peak", "cost_charge", "Cost_OnM_fixed", "cost_OnM_variable", "Total_cashflow","Investment cost: ", "NPV or LCOS"], cashflow_divided, width=1, edgecolor="white", linewidth=1)

plt.show()

plt.plot(Result_10_tries[0])
plt.plot(Result_10_tries[1])
