# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:59:18 2022

@author: jocke
"""

import numpy as np
import numpy_financial as npf


def ESS_schedule(ESS_capacity_size, ESS_power,
                 Energy_hourly_cost, Average_median_cost_day,
                 Energy_hourly_use, ESS_discharge_eff, ESS_charge_eff, Year):
    """
    Where:
    ESS_capacity_max is in kWh, max allowed kWh for that unit;
    ESS_capacity_min is in kWh, min allowed kWh for that unit;
    ESS_power in kW;
    Energy_hourly cost in pence and all hours of a year (list of 8760 hours);
    Average_median_cost_day in pence for each day of a year (list if 365 days);
    Energy_hourly_use in kWh in load demand from user for each hour in a year (list of 8760 hours)
    ESS_charge_eff and ESS_discharge_eff is given on a scale 0-1 where 1 is 100%
    
    
    It gives and 2x8760 matrix where the first array its the charge schedule of the ESS
    and the second array is the discharge schedule of the ESS

    """
    # In the schedule procuced the minimum and maximum constraints of the battery is included, the inputed value
    # for ESS capcity is changed for a max and min value in this function

    
    #Depending on the year, capacity of ESS will degrade, and cost/load demand will increase
    #Assuming linear factors (find source for this later)
    #For battery assume that after 10 years capacity is only 80% of maximum (capacity = full capacity*(year*-0,02 + 1)
    
    ESS_capacity_size = ESS_capacity_size*((Year*(-0.02))+1)
    Energy_hourly_cost = Energy_hourly_cost*((Year*0.02)+1)  #Increase energy cost by 2% each year
    Average_median_cost_day = Average_median_cost_day*((Year*0.02)+1) #Increase energy cost by 2% each year
    Energy_hourly_use = Energy_hourly_use*((Year*0.02)+1)  #Increase energy use by 2% each year, this is added to each hour element.

    # States that the max SoC is 90% of max capacity
    ESS_capacity_max = ESS_capacity_size*0.9
    # States that the min SoC is 10% of max capacity #source on this later
    ESS_capacity_min = ESS_capacity_size*0.1

    # This is as the power goes between 0-10 kw, but the number are generated from 0-100
    ESS_power = ESS_power/10

    # Matrix to store Capacity and power input/output for each hour.
    schedule_charge_discharge = np.zeros((8760, 2))
    ESS_capacity = 0  # Starts at zero energy in the ESS unit
    hour_year = 0
    

    
    
    for day_averge_cost in Average_median_cost_day:

        for hour_day in range(1, 25):

            # Checks if the average cost is higher than current (hourly) (We want to charge ESS)
            if day_averge_cost > Energy_hourly_cost[hour_year]:
                if ESS_capacity < ESS_capacity_max:             # Checks if the ESS capaicty is full or not
                    # Checks if we can charge the battery with maximum Power
                    if ESS_capacity + ESS_power*ESS_charge_eff < ESS_capacity_max:
                        # charges the ESS with its maximum power times the charge efficency
                        ESS_capacity += ESS_power*ESS_charge_eff
                        # gives an list with how charged the ESS for each hour and what happends to the ESS
                        schedule_charge_discharge[hour_year][0] = ESS_power
                    else:
                        ESS_capacity += (ESS_capacity_max - ESS_capacity)*ESS_charge_eff
                        # This is when the ESS storage is close to be full, below max rated ESS power charge
                        schedule_charge_discharge[hour_year][0] = (ESS_capacity_max - ESS_capacity)


            # Checks if the average cost is lower than current (hourly) (We want to discharge ESS)
            if day_averge_cost < Energy_hourly_cost[hour_year]:
                if ESS_capacity > ESS_capacity_min:             # Checks if the ESS capaicty is above the minimum for discharge
                    # Checks here if the energy used by the consumer is above the maximum power that can be drawn from the ESS
                    if Energy_hourly_use[hour_year] > ESS_power:
                        if ESS_capacity-ESS_capacity_min > ESS_power:  # Checks if we can discharge the battery with maximum Power
                            ESS_capacity -= ESS_power  # charges the ESS with its maximum power
                            # Sets the schedule to a negative value as it uses energy from the ESS
                            schedule_charge_discharge[hour_year][1] = ESS_power*ESS_discharge_eff
                        else:
                            ESS_capacity = ESS_capacity_min
                            # This case is when we have less than maximum power, and then uses up the last energy availbale in the ESS
                            schedule_charge_discharge[hour_year][1] = ESS_capacity*ESS_discharge_eff
                    # When the powered used by consumer is less then the maximum power by the ESS, we here then use the maximu we can but that is less than ESS power max
                    if Energy_hourly_use[hour_year] < ESS_power:
                        # Checks that the ESS have above the energy we want to discharge from it
                        if ESS_capacity > Energy_hourly_use[hour_year]:
                            ESS_capacity -= Energy_hourly_use[hour_year]
                            schedule_charge_discharge[hour_year][1] = (Energy_hourly_use[hour_year])*ESS_discharge_eff
                        # This is the case when we dont enough energy to discharge from the ESS so we take all we can take from this hour.
                        elif ESS_capacity < Energy_hourly_use[hour_year]:
                            ESS_capacity = ESS_capacity_min
                            schedule_charge_discharge[hour_year][1] = ESS_capacity*ESS_discharge_eff

            hour_year += 1  # At what hour we are in during the year

    # Returns a 8760x2 matrix where the first column is the charge schedule, and the second is the discharge scehdule
    return np.array(schedule_charge_discharge)

# -------------------------------------------------------


def Fitness_NPV(discount_rate, cashflows):
    """
    Rate is discount rate in %, 8% == 0.08
    cashflow is an array with the cashflows for each year (10)
    """
    NPV = npf.npv(discount_rate, cashflows)  #numpy financial to calculate 

    return NPV

def NPV_charging(discount_rate, cashflows_charging):
    """
    Rate is the discount rate
    """
    
    NPV_char = npf.npv(discount_rate, cashflows_charging)
    
    return NPV_char

def NPV_OM(discount_rate, cashflows_OM):
    
    NPV = npf.npv(discount_rate, cashflows_OM)
    
    return NPV

def cashflow_yearly(schedule_load, schedule_discharge, demand_cost):
    
    
    profit = np.sum(schedule_discharge*demand_cost)
    cost_charge = np.sum(schedule_load*demand_cost) #only cost for charging the unit
    cost_o_and_m = 0  #Dependent on year or hourly use per year. 
    
    cashflow_total =  profit - cost_charge - cost_o_and_m  #This is the total cashflow after a year with all calculations included
    return cashflow_total

def Roulette_wheel_selection(Population):
    """
    Takes the whole populations with the fitness values included in each solutions
    these are used to randomly choose a parent solution with unifrom distribution
    The population needs to be a np.array!!!!
    """
    
    #THIS CODES NEED TO BE REWRITTEN AS IT DOES NOT WORK IF WE HAVE NEGATIVE NPV VALUES FOR THE FITNESS
    #Therefore, we could just set them to zero so they wont ever be picked, but that would make the code biased!!!!
    
    #Based on this method : https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm
        
    
    population_fitness = np.sum(Population[:,2]) #Sum upp the fitness values from all the solutions in the population
    
    #print(population_fitness)
    rand_numb = np.random.randint(1, int(population_fitness)) #Generates a ranodom number between 1 and int of sum of the fitness values
    add_partial = 0
    counter = 0
    #print(rand_numb, "RAND NUMBER")  
    while add_partial < rand_numb:
        if add_partial < rand_numb:
            add_partial += Population[counter][2]
            #print(add_partial, counter)       
            counter += 1
            
    Parent = Population[counter-1]
       
    return Parent #Returns the array of information of the parent for that list of the population


def Tournament_selection(Population): ###This can be used for parent selection with negative fitness values.
    
    Parent = 0
    return Parent




def Fitness_max_saved(Energy_hourly_use, schedule, Energy_hourly_cost, ESS_power, ESS_capacity, ESS_capacity_cost,
                      ESS_power_cost, ESS_O_and_M_cost, Base_case_cost):
    # Fittness function to minimize the cost of energy per year including installing ESS----------
    # Swithcing this to maximise the value gained from installing ESS by taken the
    # energy saved from base case minus the cost of the ESS
    
    ESS_power = ESS_power/10    #As the values are randomly from 0-100, but we look at the power like 0-10

    ESS_total_cost = (ESS_power*ESS_power_cost) + (ESS_capacity *
                                                   ESS_capacity_cost) + (ESS_O_and_M_cost*ESS_capacity)

    # positive as discharge are negative values
    New_load_demand = power_load_59 + schedule[:, 0]

    New_case_cost = 0
    for Count_2, El_2 in enumerate(New_load_demand):
        New_case_cost += (El_2/1000)*(Energy_hourly_cost[Count_2])

    max_saved = (Base_case_cost - New_case_cost) - ESS_total_cost

    return max_saved