
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


sensitivity_capacity_cost = [ESS_capacity_cost*0.8, ESS_capacity_cost*0.9, ESS_capacity_cost, ESS_capacity_cost*1.1, ESS_capacity_cost*1.2]
sensitivity_power_cost = [ESS_power_cost*0.8, ESS_power_cost*0.9, ESS_power_cost, ESS_power_cost*1.1, ESS_power_cost*1.2]
solution_NPV_pygad = [5.723847789, 0.936398189]  # for NPV 200 gen pygad case 2
solution_LCOS_pygad = [440.7939815, 133.1479007] # for LCOS 200 gen pygad case 2


solution_NPV_FF = [0.1, 0.1]  # for NPV 200 gen ff case 2
solution_LCOS_FF = [0.1, 0.1] # for LCOS 200 gen ff case 2

Results_sensitivity_NPV_pygad = np.zeros(5)
Results_sensitivity_LCOS_pygad = np.zeros(5)
Results_sensitivity_NPV_FF = np.zeros(5)
Results_sensitivity_LCOS_FF = np.zeros(5)

for i in range(5):
    ESS_capacity_cost = sensitivity_capacity_cost[i]
    ESS_power_cost = sensitivity_power_cost[i]
    Test_NPV_pygad = fitness_func_NPV(solution_NPV_pygad)
    Test_LCOS_pygad = fitness_func_LCOS(solution_LCOS_pygad)
    
    Test_NPV_FF = fitness_func_NPV(solution_NPV_FF)
    Test_LCOS_FF = fitness_func_LCOS(solution_LCOS_FF)

    Results_sensitivity_NPV_pygad[i] = Test_NPV_pygad
    Results_sensitivity_LCOS_pygad[i] = -Test_LCOS_pygad #negative as we want to reverse the negative we used before

    Results_sensitivity_NPV_FF[i] = Test_NPV_FF
    Results_sensitivity_LCOS_FF[i] = -Test_LCOS_FF #negative as we want to reverse the negative we used before

pygad_results = [Results_sensitivity_NPV_pygad, Results_sensitivity_LCOS_pygad]
FF_results = [Results_sensitivity_NPV_FF, Results_sensitivity_LCOS_FF]
sensitivity = ["-20%", "-10%", "Base", "+10%", "+20%"]

data_sensitivity = {'Sensitivity': sensitivity,'Pygad NPV [Euro]': pygad_results[0], 'Pygad_LCOS [Euro/kWh]': pygad_results[1],
             'FF NPV [Euro]': FF_results[0], 'FF_LCOS [Euro/kWh]': FF_results[1]}

ds = pd.DataFrame(data_sensitivity, columns=['Sensitivity', 'Pygad NPV [Euro]', 'Pygad_LCOS [Euro/kWh]', 'FF NPV [Euro]',
                             'FF_LCOS [Euro/kWh]'])


ds.to_csv("Results\Sensitivity_CAPEX_case_2.csv", index = False)



X_sensitivity = ['-20%', '-10%', 'Base', '10%', '20%']

fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Change in CAPEX')
ax.set_ylabel('NPV [EURO]')
 

line1, = ax.plot(X_sensitivity, pygad_results[0], label="Case 2 GA NPV", linestyle='dashed', color = "red")
line2, = ax.plot(X_sensitivity, FF_results[0], label = "Case 2 FF NPV", linewidth = 1, color = 'lawngreen')

ax.legend(loc='right', bbox_to_anchor=(1.5, 1))

ax_2 = ax.twinx()
ax_2.set_ylabel('LCOS [EURO/kWh]')

line3, = ax_2.plot(X_sensitivity, pygad_results[1], label="Case 2 GA LCOS", linestyle='dotted', color = "cyan")
line4, = ax_2.plot(X_sensitivity, FF_results[1], label = "Case 2 FF LCOS", linestyle ='dashdot', color = 'black')

ax_2.legend(loc='right', bbox_to_anchor=(1.5, 0.8))
plt.title('Case 2: NPV and LCOS vs Change in CAPEX')
plt.savefig('Results\Sensitivity_analysis_case_2\Case_2_Sensitivity_NPV_LCOS_New.jpeg',  dpi=300, bbox_inches = "tight")


