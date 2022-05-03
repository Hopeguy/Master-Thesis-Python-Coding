# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 09:55:42 2022

@author: jocke
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:39:23 2022

@author: jocke
"""

import pandas as pd
import numpy as np
import time
import math

### Function for solving the ESS schedule ###


def ESS_schedule(ESS_capacity_size, ESS_power, 
                 Energy_hourly_cost, Average_median_cost_day, 
                 Energy_hourly_use, ESS_discharge_eff, ESS_charge_eff):
    """
    Where:
    ESS_capacity_max is in kWh, max allowed kWh for that unit;
    ESS_capacity_min is in kWh, min allowed kWh for that unit;
    ESS_power in kW;
    Energy_hourly cost in pence and all hours of a year (list of 8760 hours);
    Average_median_cost_day in pence for each day of a year (list if 365 days);
    Energy_hourly_use in kWh in load demand from user for each hour in a year (list of 8760 hours)
    ESS_charge_eff and ESS_discharge_eff is given on a scale 0-1 where 1 is 100%
    
    """
    #In the schedule procuced the minimum and maximum constraints of the battery is included, the inputed value
    # for ESS capcity is changed for a max and min value in this function
    
    ESS_capacity_max = ESS_capacity_size*0.9  #States that the max SoC is 90% of max capacity
    ESS_capacity_min = ESS_capacity_size*0.1 #States that the min SoC is 10% of max capacity #source on this later
    
   
    
    schedule_capacity = np.zeros((8760,2)) #Matrix to store Capacity and power input/output for each hour.
    ESS_capacity = 0 #Starts at zero energy in the ESS unit
    hour_year = 0
    for day_averge_cost in Average_median_cost_day:
        
        for hour_day in range(1,25):
            
            if day_averge_cost > Energy_hourly_cost[hour_year]: # Checks if the average cost is higher than current (hourly) (We want to charge ESS)
                if ESS_capacity < ESS_capacity_max:             # Checks if the ESS capaicty is full or not
                    if ESS_capacity + ESS_power*ESS_charge_eff < ESS_capacity_max: #Checks if we can charge the battery with maximum Power 
                        ESS_capacity += ESS_power*ESS_charge_eff   #charges the ESS with its maximum power times the charge efficency
                        schedule_capacity[hour_year] = ESS_power, ESS_capacity #gives an list with how charged the ESS for each hour and what happends to the ESS
                    else:
                        ESS_capacity += (ESS_capacity_max - ESS_capacity)*ESS_charge_eff
                        schedule_capacity[hour_year] = (ESS_capacity_max - ESS_capacity), ESS_capacity  #This is when the ESS storage is close to be full, below max rated ESS power charge
                        
                else:
                    schedule_capacity[hour_year, 1] = ESS_capacity #If the capacity is full, nothing happends and here we just include it to get a list of its behavior
                        
                        
            if day_averge_cost < Energy_hourly_cost[hour_year]:# Checks if the average cost is lower than current (hourly) (We want to discharge ESS)
                   if ESS_capacity > ESS_capacity_min:             # Checks if the ESS capaicty is above the minimum for discharge
                       if Energy_hourly_use[hour_year] > ESS_power:  #Checks here if the energy used by the consumer is above the maximum power that can be drawn from the ESS
                           if ESS_capacity-ESS_capacity_min > ESS_power: #Checks if we can discharge the battery with maximum Power 
                               ESS_capacity -= ESS_power   #charges the ESS with its maximum power
                               schedule_capacity[hour_year] = -ESS_power*ESS_discharge_eff, ESS_capacity #Sets the schedule to a negative value as it uses energy from the ESS
                           else:
                               ESS_capacity = ESS_capacity_min
                               schedule_capacity[hour_year] = -ESS_capacity*ESS_discharge_eff, ESS_capacity #This case is when we have less than maximum power, and then uses up the last energy availbale in the ESS
                       if Energy_hourly_use[hour_year] < ESS_power:  #When the powered used by consumer is less then the maximum power by the ESS, we here then use the maximu we can but that is less than ESS power max
                           if ESS_capacity > Energy_hourly_use[hour_year]:      #Checks that the ESS have above the energy we want to discharge from it
                               ESS_capacity -= Energy_hourly_use[hour_year]   
                               schedule_capacity[hour_year] = -(Energy_hourly_use[hour_year])*ESS_discharge_eff, ESS_capacity                           
                           elif ESS_capacity < Energy_hourly_use[hour_year]: #This is the case when we dont enough energy to discharge from the ESS so we take all we can take from this hour.
                                ESS_capacity = ESS_capacity_min
                                schedule_capacity[hour_year] = -ESS_capacity*ESS_discharge_eff, ESS_capacity
                               
                        
            
            hour_year += 1  #At what hour we are in during the year
                          

    return schedule_capacity #Returns a 8760x2 matrix where the first column is the schedule, and the second is the ESS capacity at each hour
    
######-------------------------------------------------------



def Crossover(Parent_1, Parent_2):
    """
    Parent 1 and 2 are the two solutions that are doing a cross over, and should be a
    list with both the binary capacity and power for the each parent. If you use elitism take the
    best parents together
    """
    ###For single ESS (no hybrid)
    
    #We want to swap the power between the parents, one is enough to generate the two new
    #Solutions (offspring)
    for count, bit_power in enumerate(Parent_2[1]):
        if bit_power == 1:
            save = count
            Parent_2[1][count] = 0

    
    for count2, bit_power2 in enumerate(Parent_1[1]):
        if bit_power2 == 1:
            Parent_2[1][count2] = 1
            Parent_1[1][count2] = 0
    
    Parent_1[1][save] = 1
    
    Offspring_1 = Parent_1
    Offspring_2 = Parent_2
    
    return [Offspring_1, Offspring_2]
 

#------------To test the crossover function----------------

parent_1 = [[0,1,0],[1,0,0]] #binary representation of a capacity and power for a solution
parent_2 = [[1,0,0],[0,1,0]] #the first list is the capacity, the second is the power

crossover = Crossover(Parent_1 = parent_1, Parent_2 = parent_2)

print(type(crossover))

#----------------------------------------------------------
   

#---------------Mutation function--------------------

def mutation(bit_string, mutation_chance):
    
    if np.random.rand() < mutation_chance:  #If the mutation happends we want to swap a random bit
        print("mutation happend")      #just to check that the bit swapped when we went in here
        for pos, bit in enumerate(bit_string):  #Goes through the list of binary bits
            if bit == 1:                        #When we find the active bit we want to deactive it "== 0"
                while bit_string[pos] == 1:         #Checks so we dont accidently activated the same bit again with the random function
                    if bit_string[pos] == 1:        #Check that the bit is = 1 to swap it
                        bit_string[pos] = 0         #Swaps the bit to zero "Turn it off"
                        bit_string[np.random.randint(0, len(bit_string))] = 1       #Randomly starts another bit (putting it == 1)
                break        #When we have made a new bit that was not the old one activated we break the loop and return the new binary code list
                    
        
    return bit_string       #Returns the new binary code list
                    
#-----------To test the mutaion function--------                                     
mutation_string = mutation(bit_string = [0, 1, 0], mutation_chance = 0.5)               

print(mutation_string)                
            
#-----------------------------------------------
        


def Fitness_max_saved(Load_demand, schedule, Energy_hourly_cost, ESS_power, ESS_capacity, ESS_capacity_cost,
            ESS_power_cost, ESS_O_and_M_cost, Base_case_cost):
### Fittness function to minimize the cost of energy per year including installing ESS----------
### Swithcing this to maximise the value gained from installing ESS by taken the 
### energy saved from base case minus the cost of the ESS

    
    ESS_total_cost = (ESS_power*ESS_power_cost) + (ESS_capacity*ESS_capacity_cost) + (ESS_O_and_M_cost*ESS_capacity)
    
    New_load_demand = power_load_59 + Schedule[:,0] #positive as discharge are negative values
    
    New_case_cost = 0
    for Count_2, El_2 in enumerate(New_load_demand):
        New_case_cost += (El_2/1000)*(Energy_hourly_cost[Count_2])
    
    max_saved = (Base_case_cost - New_case_cost) - ESS_total_cost
    
    return max_saved




def Fitness_NPV():
    
    
    
    
    NPV = 0
    
    return NPV

            


def Genetic_algorithm():
    
    
    
    
    best_solution = 0
    return best_solution





##----------------------------------



### Input values for solution matrix ###

Battery_size = list(range(1, 101)) #kWh
Battery_power = (list(range(1, 101))) #kW

for count, power in enumerate(Battery_power):
    Battery_power[count] = power*0.1


Battery_size_binary = np.zeros(100, dtype=int)
Battery_power_binary = np.zeros(100, dtype = int)


#----------Gets the average cost for each day, and the hourly cost at each hour during the year--------

Electricity_price_read_oslo = np.genfromtxt("os-eur17.csv", delimiter = ",") #Prices in EUR/MWh
El_cost_year = []
El_cost_average_day = []
for i in range(365):
    for k in Electricity_price_read_oslo[i][0:24]:
         El_cost_year.append(k)
    El_cost_average_day.append(Electricity_price_read_oslo[i][24])

###-------------Read load data for each hour of a year of house 59---------
El_data_read = pd.read_csv("home59_hall687_sensor1506c1508_electric-mains_electric-combined.csv",
                      header = None)

El_data_59 = El_data_read[1][2038:8760+2038] #Average W used for that hour (giving the Wh) from second of january

power_load_59 = []
for i in El_data_59:
    power_load_59.append(i/1000)   #in kWh

###--------------------------------------------------------------------------



#---------------Input values (non changable)---------------------------------------'
#------For Schedule inputs-----------

Energy_hourly_cost = El_cost_year
Average_median_cost_day = El_cost_average_day
Energy_hourly_use = power_load_59
ESS_charge_eff = 0.9
ESS_discharge_eff = 0.9


#For schedule the matrix containing the batterys different power/capacity:
    
Battery_size = list(range(1, 101)) #kWh
Battery_power = list(range(1, 101)) #kW

#Important to note that the maximum SoC for the battery is calculated in the schdule function
#Only import is the Total max size that is also used for calculating the cost

#------For NPV inputs -------------
Lifetime_battery = 10 # in years
interest_rate = 0.07 # 7 percent
ESS_capital_cost_per_kWh = 100 #in pound per kWh
ESS_capital_cost_per_kW = 20
ESS_O_and_M_per_kWh = 1 #in Pound per kWh

###-------------Test schedule function-------

start = time.time()

Schedule = ESS_schedule(ESS_capacity_size = 1,ESS_power = 0.1, Energy_hourly_cost = Energy_hourly_cost, 
                        Average_median_cost_day = Average_median_cost_day, 
                        Energy_hourly_use = Energy_hourly_use, ESS_charge_eff = ESS_charge_eff,  
                        ESS_discharge_eff = ESS_discharge_eff)

end = time.time()
print(end - start)


### Calculate Base case cost ####
Base_case_cost = 0 #  in  pence

for Count, El in enumerate(power_load_59):
    Base_case_cost += (El)*(El_cost_year[Count]) #Divide by 1000 to get how many kWh
        

New_load_demand = power_load_59 + Schedule[:,0] #positive as discharge are negative values
New_case_cost = 0
for Count_2, El_2 in enumerate(New_load_demand):
    New_case_cost += (El_2/1000)*(El_cost_year[Count_2])
    
    
Saving = Base_case_cost - New_case_cost  #The difference is the savings in money by installing ESS
print(Saving)    

Saving_2 = Fitness_max_saved(Load_demand = Energy_hourly_use, schedule = Schedule, Energy_hourly_cost=Energy_hourly_cost,
                             ESS_power = 0.1, ESS_capacity = 1, ESS_capacity_cost = ESS_capital_cost_per_kWh,
                             ESS_power_cost = ESS_capital_cost_per_kW, ESS_O_and_M_cost = ESS_O_and_M_per_kWh,
                             Base_case_cost = Base_case_cost)
print(Saving_2)