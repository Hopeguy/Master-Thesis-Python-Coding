
import pandas as pd
import numpy as np
import numpy_financial as npf
import functions as fun
import time
import pygad
import matplotlib.pyplot as plt

def fitness_func_LCOS(solution):
    global cashflow_divided
    global ESS_capacity, ESS_power
    global Schedule_sum
    global Schedule
    global Schedule_capacity
    global Cost_yearly
    global ESS_capacity_year
    ESS_capacity, ESS_power = solution[0], solution[1]
    CAPEX = ((ESS_capacity_cost*ESS_capacity) + (ESS_power*ESS_power_cost)) 
    Cost_yearly = []
    Energy_yearly = []
    Schedule_sum = np.zeros(2)
    cashflow_divided = np.zeros(4)
    ESS_capacity_year = 0 #Starts with zero energy in the storage
    Cost_yearly_combined = []
    
    for year in range(10): # starts at year 1 and includes year 10 as the lifetime of ESS is 10 years (battery)
        
        Schedule = fun.ESS_schedule(ESS_capacity_size=ESS_capacity, ESS_power=ESS_power,  #Schedule function see function for more details
                                        Energy_hourly_cost=Energy_hourly_cost,
                                        Average_median_cost_day=Average_median_cost_day,
                                        Energy_hourly_use=Energy_hourly_use,
                                        ESS_discharge_eff=ESS_discharge_eff, ESS_charge_eff=ESS_charge_eff, Year = year, ESS_capacity_prev_year = ESS_capacity_year)

            
       
        New_schedule = Schedule[0] #saves the schedule for this year
        ESS_capacity_year = Schedule[1]  #Inputs the preveious years ess capacity to next years
        Schedule_capacity = Schedule[2]     #Gives the array with the capacity for each hour during this year.

        Cost_yearly = (fun.Cost_yearly_LCOS(schedule_load = New_schedule[:,0], schedule_discharge = New_schedule[:,1], demand_cost = Energy_hourly_cost, #Gives the yearly cost related with the schedule produced
                             Fixed_O_and_M_cost = Fixed_ESS_O_and_M_cost, Variable_O_and_M_cost = Variable_ESS_O_and_M_cost, ESS_power = ESS_power))
        Cost_yearly_combined.append(Cost_yearly[0])
        Schedule_sum[0] = np.sum(New_schedule[:, 0])  #Charge shcedule summed up kWh
        Schedule_sum[1] = np.sum(New_schedule[:, 1])  #Discharge summed up in kWh
        
        for count, i in enumerate(Cost_yearly[1]): #Charge, fixed OnM, Variable OnM, Combinded
            cashflow_divided[count] += i
        
        Energy_yearly.append(np.sum(New_schedule[:,1]))

    fitness_LCOS = fun.Fittnes_LCOS(discount_rate = Discount_rate, CAPEX = CAPEX, Yearly_cost = Cost_yearly_combined, Yearly_energy_out = Energy_yearly)
    
    return -fitness_LCOS #negative as the function want to maximize but we want the lowest value for LCOS in Euro / kWh


def fitness_func_NPV(solution):
    """
    Returns the NPV value (used as fitness value in GA)
    """
    global cashflow_divided
    global ESS_capacity, ESS_power
    global Schedule_sum
    global Schedule
    global Schedule_capacity
    global ESS_capacity_year
    ESS_capacity, ESS_power = solution[0], solution[1]
    
    cashflow_each_year = [-((ESS_capacity_cost*ESS_capacity) + (ESS_power*ESS_power_cost))] #First Year is just capex cost and negative as it is a cost
    cashflow_divided = np.zeros(7)
    Schedule_sum = np.zeros(2)
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
        Schedule_capacity = Schedule[2]
        Peak_diff = fun.Peak_diff(Electricty_usage_pre_schedule = Energy_hourly_use, Schedule = Schedule[0])
        Cashflow_yearly = fun.cashflow_yearly_NPV(schedule_load = New_schedule[:, 0], schedule_discharge = New_schedule[:,1], demand_cost = Energy_hourly_cost,
                                                        Variable_O_and_M_cost = Variable_ESS_O_and_M_cost, Fixed_O_and_M_cost = Fixed_ESS_O_and_M_cost,
                                                        ESS_power = ESS_power, Peak_diff = Peak_diff, Peak_diff_cost = Peak_cost)
        #print(Cashflow_yearly[1])
        for count, i in enumerate(Cashflow_yearly[1]):
            cashflow_divided[count] += i
        Schedule_sum[0] = np.sum(New_schedule[:, 0])  #Charge shcedule summed up kWh
        Schedule_sum[1] = np.sum(New_schedule[:, 1])  #Discharge summed up in kWh
        
        cashflow_each_year.append(Cashflow_yearly[0])
    fitness = fun.Fitness_NPV(discount_rate = Discount_rate, cashflows = cashflow_each_year)
    return fitness 

    
Electricity_price_read = np.genfromtxt(
    "sto-eur17.csv", delimiter=",")  # Prices in EUR/MWh
El_cost_year = []
El_cost_average_day = []

for i in range(365):
    for k in Electricity_price_read[i][0:24]:
        El_cost_year.append((k/1000)*1.1) #Prices in Euro/kWh, by dividing by 1000, times 1.1 to get 2022 euro values
          
    El_cost_average_day.append(((Electricity_price_read[i][24])/1000)*1.1)  #Prices in Euro/kWh that is why we are dividing by 1000, times 1.1 to get 2022 values


# --------------------Read load data for both Electricity and Heating--------
Load_data_read = pd.read_csv("Load_data_electricit_heating_2017.csv", header=0) #Takes values from January, Empty data in 7976 set to 0
Electricity_load_pd = Load_data_read["Electricty [kW]"]

Electricity_load = np.zeros(8760)

for count, i in enumerate(Electricity_load_pd):
    Electricity_load[count] = i

# --------------------------------------------------------------------------

# ------For Schedule inputs-----------

Energy_hourly_cost = np.array(El_cost_year)
Average_median_cost_day = np.array(El_cost_average_day)
Energy_hourly_use = np.array(Electricity_load)
ESS_charge_eff = 0.9
ESS_discharge_eff = 0.9

# Important to note that the maximum SoC for the battery is calculated in the schedule function

# ------For NPV/LCOE inputs -------------
Lifetime_battery = 10  # in years
sensitivity_factor = 0.8
ESS_capacity_cost = 389.2#*sensitivity_factor   # in Euro(2022) per kWh (CAPEX) all cost included
ESS_power_cost = 148.8#*sensitivity_factor  # in Euro(2022) per kW (all cost included)
Fixed_ESS_O_and_M_cost = 4.19  # in Euro(2022) per kW-year
Variable_ESS_O_and_M_cost = 0.488/1000 # in Euro(2022) per kWh-year 
Discount_rate = 0.08 #8 percent   
Peak_cost = 5.92/1.1218 #5.92 dollar (2022) per kW (max per month) change to euro: 1 euro is 1.1218 USD january 1 2022


solution_NPV_pygad = [5.723847789, 0.936398189]  # for NPV 200 gen pygad case 2
solution_LCOS_pygad = [440.7939815, 133.1479007] # for LCOS 200 gen pygad case 2
solution_NPV_FF = [0.1, 0.1]  # for NPV 200 gen ff case 2
solution_LCOS_FF = [0.1, 0.1] # for LCOS 200 gen ff case 2

solution_NPV_pygad_case_3 = [5.723847789, 0.936398189]  # for NPV 200 gen pygad case 2
solution_LCOS_pygad_case_3 = [440.7939815, 133.1479007] # for LCOS 200 gen pygad case 2
solution_NPV_FF_case_3 = [0.1, 0.1]  # for NPV 200 gen ff case 2
solution_LCOS_FF_case_3 = [0.1, 0.1] # for LCOS 200 gen ff case 2


# For cost divided NVP using best result from 200 generation (highest NPV)


case_2_data_NPV = pd.read_csv('Results\Pygad_case_2_ESS_NPV\ESS_power_NPV_etc\Pygad_case_2_NPV_ESS_200_gen.csv') #GA
Case_2_FF_data_NPV = pd.read_csv('Results\Firefly_case_2_ESS_NPV\ESS_power_NPV_etc\Firefly_case_2_ESS_NPV_200_gen.csv') #FF

#Sort the dataframe so at index 1 is the solution with the best fitness value (highes NPV)
case_2_data_NPV.sort_values('fitness_function', ascending=False, inplace=True, ignore_index=True)
Case_2_FF_data_NPV.sort_values('fitness_function', ascending=False, inplace=True, ignore_index=True)

Solution_case_2_data_NPV_GA = [case_2_data_NPV["ESS_capacity"][0], case_2_data_NPV["ESS_power"][0]] #capacity, power
Solution_case_2_data_NPV_FF = [Case_2_FF_data_NPV["ESS_capacity"][0], Case_2_FF_data_NPV["ESS_power"][0]] #capacity, power




Case2_ALL = [Solution_case_2_data_NPV_GA, Solution_case_2_data_NPV_FF]


#Calculating for case 2 firstly

Result_divided = [[],[],[],[],[],[],[],[],[],[],[],[]]
for i in Case2_ALL:
    ESS_capacity, ESS_power = i[0], i[1]
    
    Result_NPV = fitness_func_NPV(i)

    

    cost_investment = -((ESS_capacity_cost*ESS_capacity) + (ESS_power*ESS_power_cost))
    cashflow_divided[6] = cost_investment  #Adds the investment cost
    Result_divided[0].append(ESS_capacity) #Capacity
    Result_divided[1].append(ESS_power) #Power
    Result_divided[2].append(Result_NPV) #fittness function positive when looking for NPV
    Result_divided[3].append(cashflow_divided[0]) #profit from selling kWh
    Result_divided[4].append(cashflow_divided[1])  #profit from Peak_kW
    Result_divided[5].append(cashflow_divided[2])  #cost from chargin
    Result_divided[6].append(cashflow_divided[3])  #cost OnM fixed
    Result_divided[7].append(cashflow_divided[4])  #Cost OnM Variable
    Result_divided[8].append(cashflow_divided[5])  #Cashflow_total
    Result_divided[9].append(cashflow_divided[6]) #cost investment
    Result_divided[10].append(Schedule_sum[0]) #Charge energy to BESS
    Result_divided[11].append(Schedule_sum[1]) #Discharge energy from BESS

print(Result_divided)
raw_data = {'Algorithm':["GA","FF"],
    'ESS_power': Result_divided[1],
                'ESS_capacity': Result_divided[0],
                'fitness_function': Result_divided[2],
                'profit_kWh': Result_divided[3],
                'profit_peak_kW': Result_divided[4],
                'cost_charge': Result_divided[5],
                'cost_O_n_M_fixed': Result_divided[6],
                'cost_O_n_m_variable': Result_divided[7],
                'Cashflow_total': Result_divided[8],
                'Cost_investment': Result_divided[9],
                'Summed_charge_kWh': Result_divided[10],
                'Summed_Discharge_kWh': Result_divided[11]}

df = pd.DataFrame(raw_data, columns = ['Algorithm', 'ESS_power', 'ESS_capacity', 'fitness_function',
                                           'profit_kWh', 'profit_peak_kW',
                                           'cost_charge', 'cost_O_n_M_fixed', 'cost_O_n_m_variable', 'Cashflow_total',
                                           'Cost_investment', 'Summed_charge_kWh', 'Summed_Discharge_kWh'])

df.to_csv("Results\Divided_cost_results_200_gen\Case_2_NPV_Divided_200_gen.csv", index=False, )