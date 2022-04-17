# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:59:18 2022

@author: jocke
"""

import numpy as np
import numpy_financial as npf



def ESS_schedule(ESS_capacity_size, ESS_power,
                 Energy_hourly_cost, Average_median_cost_day,
                 Energy_hourly_use, ESS_discharge_eff, ESS_charge_eff, Year, ESS_capacity_prev_year):
    """
    Where:
    ESS_power in kW;
    Energy_hourly cost in euro and all hours of a year (list of 8760 hours);
    Average_median_cost_day in euro for each day of a year (list if 365 days);
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
    
    ESS_capacity_size = ESS_capacity_size*((Year*(-0.02))+1) #Reduction in energy capacity after a year of usage
    Energy_hourly_cost = Energy_hourly_cost*((Year*0.02)+1)  #Increase energy cost by 2% each year
    Average_median_cost_day = Average_median_cost_day*((Year*0.02)+1) #Increase energy cost by 2% each year
    Energy_hourly_use = Energy_hourly_use*((Year*-0.0227)+1)  #Electricity 2.27% and thermal 1.8% decrease yearly.

    # States that the max SoC is 90% of max capacity
    ESS_capacity_max = ESS_capacity_size*0.9
    # States that the min SoC is 10% of max capacity #source on this later
    ESS_capacity_min = ESS_capacity_size*0.1

    # Matrix to store Capacity and power input/output for each hour.
    schedule_charge_discharge = np.zeros((8760, 2))
    ESS_capacity = ESS_capacity_prev_year  # Starts at what value one inputs, but should be that it saved the energy from last year.
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

    # Returns a 8760x2 matrix where the first column is the charge schedule, and the second is the discharge scehdule, third is the capacity at the end of the year
    #The discharge and charge is how much kWh that is charged or discharge at each hour
    return np.array(schedule_charge_discharge), ESS_capacity

# -------------------------------------------------------

def Fittnes_LCOS(discount_rate, CAPEX, Yearly_cost, Yearly_energy_out):
    """'
    Where intreset rate is in 0.08 for 8%
    CAPEX is in EURO
    Yearly cost is an array with each years total cost (O&M, charge)
    Yearly energy out is an array with each year total output from ESS
    """
    cost_intreset = 0
    for year, c in enumerate(Yearly_cost):
       cost_intreset += c/((1+discount_rate)**(year+1)) #as the enumerate counting start at 0, we add 1, at year zero we only have CAPEX
       
       
    energy_interset = 0
    for year, w in enumerate(Yearly_energy_out):
       energy_interset += w/((1+discount_rate)**(year+1))

    LCOS = (CAPEX + cost_intreset)/energy_interset
    
    return LCOS

def Fitness_NPV(discount_rate, cashflows):
    """
    Rate is discount rate in %, 8% == 0.08
    cashflow is an array with the cashflows for each year (10)
    """
    NPV = npf.npv(discount_rate, cashflows)  #numpy financial to calculate 

    return NPV



def cashflow_yearly_NPV(schedule_load, schedule_discharge, demand_cost, Fixed_O_and_M_cost, Variable_O_and_M_cost): #Gives the profits after all yearly costs and profits
    
    
    profit = np.sum(schedule_discharge*demand_cost)
    cost_charge = np.sum(schedule_load*demand_cost) #only cost for charging the unit
    cost_o_and_m = np.sum(schedule_discharge*Fixed_O_and_M_cost) + (np.sum(schedule_discharge*Variable_O_and_M_cost)/1000)  #Dependent on year or hourly use per year. 
    
    cashflow_total =  profit - cost_charge - cost_o_and_m  #This is the total cashflow after a year with all calculations included
    return cashflow_total

def Cost_yearly_LCOS(schedule_load, schedule_discharge, demand_cost, Fixed_O_and_M_cost, Variable_O_and_M_cost): #Gives the profits after all yearly costs and profits
    

    cost_charge = np.sum(schedule_load*demand_cost) #only cost for charging the unit
    cost_o_and_m = np.sum(schedule_discharge*Fixed_O_and_M_cost) + (np.sum(schedule_discharge*Variable_O_and_M_cost)/1000)  #Dependent on year or hourly use per year. 
    
    cost_yearly = cost_charge + cost_o_and_m  #This is the total cashflow after a year with all calculations included
   
    return cost_yearly
