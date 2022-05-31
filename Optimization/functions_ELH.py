# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:59:18 2022

@author: jocke
"""

import numpy as np
import numpy_financial as npf
from sympy import N



def ESS_schedule(ESS_capacity_size, ESS_power,
                 Energy_hourly_cost, Average_median_cost_day,
                 Energy_hourly_use, ESS_discharge_eff, ESS_charge_eff, Year, ESS_capacity_prev_year):
    """
    Where:
    ESS_power in kW;
    ESS_capacity = kWh
    Energy_hourly cost in euro and all hours of a year (list of 8760 hours);
    Average_median_cost_day in euro for each day of a year (list if 365 days);
    Energy_hourly_use in kWh in load demand from user for each hour in a year (list of 8760 hours)
    ESS_charge_eff and ESS_discharge_eff is given on a scale 0-1 where 1 is 100%
    
    
    It gives and 2x8760 matrix where the first array its the charge schedule of the ESS
    and the second array is the discharge schedule of the ESS

    """
    # In the schedule procuced the minimum and maximum constraints of the battery is included, the inputed value
    # for ESS capcity is changed for a max and min value in this function

    
    #Depending on the year, capacity of ESS will degrade
    #cost/load demand wont increase/decrease over the years as we are looking into a small house/set of houses (assumption)

    ESS_capacity_size = ESS_capacity_size*((Year*(-0.01612))+1) #a reduction of 1,612% each year. totaling in 15% at year 10
    #Energy_hourly_cost = Energy_hourly_cost*((Year*0.02)+1)  #Increase energy cost by 2% each year (from calculations no future energy price increase can be seen)
    #Average_median_cost_day = Average_median_cost_day*((Year*0.02)+1) #Increase energy cost by 2% each year (from calculations no future energy price increase can be seen)
    #Energy_hourly_use = Energy_hourly_use*((Year*-0.0227)+1)  #Electricity 2.27% and thermal 1.8% decrease yearly. %This might not be interesting and could be excluded as a single house wont change for a single home


    
    ESS_capacity_max = ESS_capacity_size*1 #Moongrid source 2020 #read section 3.5
    ESS_capacity_min = ESS_capacity_size*0.2 #80%  read seciton 3.5, source moongrid 2020

    # Matrix to store Capacity and power input/output for each hour.
    schedule_charge_discharge = np.zeros((8760, 2))
    schedule_capacity = np.zeros(8760)
    if ESS_capacity_size < ESS_capacity_prev_year: # Sets the starting capacity in the ESS to what was last year, and if the max capacity have become smaller than what was stored last year it sets it to maximum.
        ESS_capacity = ESS_capacity_size
    else:
        ESS_capacity = ESS_capacity_prev_year
    
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
                        schedule_charge_discharge[hour_year][0] = (ESS_capacity_max - ESS_capacity)
                        ESS_capacity = ESS_capacity_max
                        
                        # This is when the ESS storage is full, below max rated ESS power charge
                        


            # Checks if the average cost is lower than current (hourly) (We want to discharge ESS)
            if day_averge_cost < Energy_hourly_cost[hour_year]:
                if ESS_capacity > ESS_capacity_min:             # Checks if the ESS capaicty is above the minimum for discharge
                    # Checks here if the energy used by the consumer is above the maximum power that can be drawn from the ESS
                    if Energy_hourly_use[hour_year] > ESS_power:
                        if ESS_capacity-ESS_capacity_min > ESS_power:  # Checks if we can discharge the battery with maximum Power
                            ESS_capacity -= ESS_power  # charges the ESS with its maximum power
                            # Sets the schedule to a negative value as it uses energy from the ESS
                            schedule_charge_discharge[hour_year][1] = ESS_power*ESS_discharge_eff
                        else: #(ESS_capacity-ESS_capacity_min) < ESS_power:
                            schedule_charge_discharge[hour_year][1] = (ESS_capacity - ESS_capacity_min)*ESS_discharge_eff
                            ESS_capacity = ESS_capacity_min # We reduce the capacity to the minimum value (we use up what is left)
                            
                            # This case is when we have less than maximum power, and then uses up the last energy availbale in the ESS
                            
                            
                    # When the powered used by consumer is less then the maximum power by the ESS, we here then use the maximu we can but that is less than ESS power max
                    if Energy_hourly_use[hour_year] < ESS_power:
                        # Checks that the ESS have above the energy we want to discharge from it
                        if (ESS_capacity-ESS_capacity_min) > (Energy_hourly_use[hour_year]/ESS_discharge_eff):
                            ESS_capacity -= Energy_hourly_use[hour_year]/ESS_discharge_eff #We remove the capacity equal to what the user want divided by the eff in order to give them exactly what they need.
                            schedule_charge_discharge[hour_year][1] = (Energy_hourly_use[hour_year]) #We "sell" the amount they want to use
                        # This is the case when we dont enough energy to discharge from the ESS so we take all we can take from this hour.
                        else: #(ESS_capacity-ESS_capacity_min) < Energy_hourly_use[hour_year]:
                            schedule_charge_discharge[hour_year][1] = (ESS_capacity - ESS_capacity_min)*ESS_discharge_eff
                            ESS_capacity = ESS_capacity_min
                         
                            
            schedule_capacity[hour_year] = (ESS_capacity)
            hour_year += 1  # At what hour we are in during the year

    # Returns a 8760x2 matrix where the first column is the charge schedule, and the second is the discharge scehdule, third is the capacity at the end of the year
    #The discharge and charge is how much kWh that is charged or discharge at each hour

    return np.array(schedule_charge_discharge), ESS_capacity, schedule_capacity

# -------------------------------------------------------

def Residual_value_ELH(Interest_rate, ELH_power_cost, ELH_power, Lifetime_ELH, project_lifetime):
    ELH_cost = ELH_power*ELH_power_cost
    Monthly_value = npf.pmt(pv = ELH_cost, nper = Lifetime_ELH, rate = Interest_rate)
    Resiudal_value = Monthly_value*(abs(Lifetime_ELH-project_lifetime))
    return Resiudal_value

def Peak_diff(Electricty_usage_pre_schedule, Schedule):
    """
    Electicity_usage_pre_schedule is the electricity usage before an battery have been installed
    Schedule is the charge and discharge schedule for that year,
    the Value will return the peak difference for each month that then can be used as a profit for installing the Battery
    """

    New_electricity_usage_with_discharge = np.subtract(Electricty_usage_pre_schedule, Schedule[:,1]) #Schedule 1 is the discharge schedule
    New_electricity_usage_with_discharge_and_charge = np.add(New_electricity_usage_with_discharge, Schedule[:,0]) #if we want to include the chargin also to the calculations


    Monthly_max_pre = np.zeros(12)
    Monthly_max_after = np.zeros(12)
    for count in range(12):

        Monthly_max_pre[count] = np.max(Electricty_usage_pre_schedule[count*730:(count+1)*730])
        Monthly_max_after[count] = np.max(New_electricity_usage_with_discharge_and_charge[count*730:(count+1)*730])
    
    Monthly_peak_diff = np.subtract(Monthly_max_pre, Monthly_max_after) #subtracts the discharge and adds the charge from the pre schedule with the schedule
    #print(Monthly_peak_diff)

    return Monthly_peak_diff #Array with 12 values (on for each month of the year)

def Fittnes_LCOS(discount_rate, CAPEX, Yearly_cost, Yearly_energy_out):
    """'
    Where discount rate is in 0.08 for 8%
    CAPEX is in EURO 2022
    Yearly cost is an array with each years total cost (O&M, charge)
    Yearly energy out is an array with each year total output from ESS
    """
    cost_intrest = 0
    
    for year, c in enumerate(Yearly_cost):
       cost_intrest += c/((1+discount_rate)**(year+1)) #as the enumerate counting start at 0, we add 1, at year zero we only have CAPEX
       
       
    energy_intrest = 0
    for year, w in enumerate(Yearly_energy_out):
       energy_intrest += w/((1+discount_rate)**(year+1))

    LCOS = (CAPEX + cost_intrest)/energy_intrest
    
    return LCOS #returns LCOS in Euro/kWh

def Fitness_NPV(discount_rate, cashflows):
    """
    Rate is discount rate in %, 8% == 0.08
    cashflow is an array with the cashflows for each year (10)
    """
    NPV = npf.npv(discount_rate, cashflows)  #numpy financial to calculate 

    return NPV #In Euro



def cashflow_yearly_NPV(schedule_load, schedule_discharge, demand_cost, Fixed_O_and_M_cost,
                        Variable_O_and_M_cost, ESS_power, ELH_power, ELH_OPEX, Gas_cost,
                        Heating_demand_after_ELH, Heating_demand_pre, Peak_diff, Peak_diff_cost, electricity_load_ELH): #Gives the profits after all yearly costs and profits

    profit_kWh = abs(np.sum(schedule_discharge*demand_cost)) #profit made from discharge of the ESS at higher electricity cost
    profit_peak_kW = abs(np.sum(Peak_diff*Peak_diff_cost)) #profit made form peak difference monthly using the ESS instead of using only grid
    cost_charge = abs(np.sum(schedule_load*demand_cost)) #only cost for charging the unit
    cost_o_and_m_fixed = abs(ESS_power*Fixed_O_and_M_cost)   #Dependent on year or hourly use per year. 
    cost_o_and_m_variable = abs(np.sum(schedule_discharge*Variable_O_and_M_cost))
    cost_OPEX_ELH = ELH_power*ELH_OPEX #Opex of Electrical heater yearly
    Saved_cost_heating = ((np.sum(Heating_demand_pre) - np.sum(Heating_demand_after_ELH))*Gas_cost) #Saved money on gas heating
    Heating_electricity_cost = np.sum(electricity_load_ELH*demand_cost) #Cost of using electricty to heat instead
    profit_saved_heating_total = Saved_cost_heating - Heating_electricity_cost #Combined saved money on using electrical heating instead of gas
    cashflow_total =  profit_kWh + profit_peak_kW + profit_saved_heating_total - cost_charge - cost_o_and_m_fixed - cost_o_and_m_variable - cost_OPEX_ELH#This is the total cashflow after a year with all calculations included
    Divided_cost_profit = [profit_kWh, profit_peak_kW, profit_saved_heating_total, Saved_cost_heating, (-Heating_electricity_cost), (-cost_charge), (-cost_o_and_m_fixed), (-cost_o_and_m_variable), (-cost_OPEX_ELH), cashflow_total]
    return [cashflow_total, Divided_cost_profit]


def Cost_yearly_LCOS(schedule_load, schedule_discharge, demand_cost, Fixed_O_and_M_cost,
                 Variable_O_and_M_cost, ESS_power, ELH_power, ELH_OPEX): #Gives the profits after all yearly costs and profits
    
    cost_charge = np.sum(schedule_load*demand_cost) #only cost for charging the unit
    cost_o_and_m_fixed = (ESS_power*Fixed_O_and_M_cost) #Dependent on year or hourly use per year. 
    cost_o_and_m_variable = np.sum(schedule_discharge*Variable_O_and_M_cost)
    #cost_o_and_m_ELH = (ELH_power*ELH_OPEX) #not included
    cost_yearly = cost_charge + cost_o_and_m_variable + cost_o_and_m_fixed #This is the total cost after a year with all parts included
    
    Cost_divided = [cost_charge, cost_o_and_m_fixed, cost_o_and_m_variable, cost_yearly]
    
    return [cost_yearly, Cost_divided]

def Residual_value_ELH(Interest_rate, ELH_power_cost, ELH_power, Lifetime_ELH, project_lifetime):
    ELH_cost = ELH_power*ELH_power_cost
    Monthly_value = npf.pmt(pv = ELH_cost, nper = Lifetime_ELH, rate = Interest_rate)
    Resiudal_value = Monthly_value*(abs(Lifetime_ELH-project_lifetime))
    return Resiudal_value
