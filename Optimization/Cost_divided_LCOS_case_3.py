
import pandas as pd
import numpy as np
import numpy_financial as npf
import functions_ELH as fun
import time
import pygad
from operator import add

def Electricity_heater_load(Power, Heating_load):
    """"
    Power in kW
    Heating load in kWh (An array with 8760 hours)
    Efficency set to 95% (Source)
    Output is the new lower heating load that has been taken care of with the electrical heater (array of 8760 hours)
    Electrical load from the electrical heater (array of 8760 hours)
    """
    
    Efficency = 0.95 # Depending on source but assumed to be 95% (Source  "UKSupplyCurve") #between 90-100%
    Electricity_load = np.zeros(8760)
    New_heating_load = np.zeros(8760)
    for count, load in enumerate(Heating_load): #goes through the array with the load demand
        if load < Power:                        #if the load is less then the power of the electrical heater
            Electricity_load[count] = load/Efficency    #Electricity load is increase by the load divided by the efficency
            New_heating_load[count] = 0                 #as the load was lower then the power zero heating load is left this hour
        elif load > Power:
            Electricity_load[count] = Power/Efficency   # When the load is higher than the Power of the electrical heater, the new electricty this hour is the power divided by the efficency
            New_heating_load[count] = load - Power #The heat load this hour is the load minus the power that was removed by the electrical heater

    return Electricity_load, New_heating_load

def fitness_func_LCOS(solution):
    global cashflow_divided
    global ESS_capacity, ESS_power
    global Schedule_sum
    global Schedule
    global Schedule_capacity
    global Cost_yearly
    global ESS_capacity_year
    global Energy_hourly_use
    global Electricity_heating_load
    global Energy_yearly

    ESS_capacity, ESS_power, ELH_power = solution[0], solution[1], solution[2]
    CAPEX=((ESS_capacity_cost*ESS_capacity) + \
           (ESS_power*ESS_power_cost)) #capex for ELH is not included
    Energy_yearly = []
    Schedule_sum = np.zeros(2)
    cashflow_divided = np.zeros(4)
    ESS_capacity_year = 0 #Starts with zero energy in the storage
    Cost_yearly_combined = []
    # array first is the electricity load from ELH, and second array is the new Heating load
    Electricity_heating_load = Electricity_heater_load(
        Power=ELH_power, Heating_load=Heating_hourly_use)
    # adds the electricity load with the ELH electricity load
    Energy_hourly_use = Electricity_load + Electricity_heating_load[0]

    # starts at year 1 and includes year 10 as the lifetime of ESS is 10 years (battery)
    for year in range(10): # starts at year 1 and includes year 10 as the lifetime of ESS is 10 years (battery)
        
        Schedule = fun.ESS_schedule(ESS_capacity_size=ESS_capacity, ESS_power=ESS_power,  #Schedule function see function for more details
                                        Energy_hourly_cost=Energy_hourly_cost,
                                        Average_median_cost_day=Average_median_cost_day,
                                        Energy_hourly_use=Energy_hourly_use,
                                        ESS_discharge_eff=ESS_discharge_eff, ESS_charge_eff=ESS_charge_eff, Year = year, ESS_capacity_prev_year = ESS_capacity_year)

            
       
        New_schedule = Schedule[0] #saves the schedule for this year
        ESS_capacity_year = Schedule[1]  #Inputs the preveious years ess capacity to next years
        Schedule_capacity = Schedule[2]     #Gives the array with the capacity for each hour during this year.

        Cost_yearly = (fun.Cost_yearly_LCOS(schedule_load = New_schedule[:,0], schedule_discharge = New_schedule[:,1],
                            demand_cost = Energy_hourly_cost, Fixed_O_and_M_cost = Fixed_ESS_O_and_M_cost,
                            Variable_O_and_M_cost = Variable_ESS_O_and_M_cost, ESS_power = ESS_power,
                            ELH_OPEX= ELH_OPEX, ELH_power=ELH_power)) #Gives the yearly cost related with the schedule produced

        Cost_yearly_combined.append(Cost_yearly[0]) #Only total cost is appended for each year
        Schedule_sum[0] = np.sum(New_schedule[:, 0])  #Charge shcedule summed up kWh
        Schedule_sum[1] = np.sum(New_schedule[:, 1])  #Discharge summed up in kWh
        
        for count, i in enumerate(Cost_yearly[1]): #Charge, fixed OnM, Variable OnM, Combinded
            
            cashflow_divided[count] += i
        
        Energy_yearly.append(np.sum(New_schedule[:,1]))

    fitness_LCOS = fun.Fittnes_LCOS(discount_rate = Discount_rate, CAPEX = CAPEX, Yearly_cost = Cost_yearly_combined, Yearly_energy_out = Energy_yearly)
    return fitness_LCOS  # returns LCOS in Euro/kWh

def callback_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])
    


#--------------Read price data for electricity and sets it up into hourly and daily average------------

Electricity_price_read = np.genfromtxt("sto-eur17.csv", delimiter=",")  # Prices in EUR/MWh
El_cost_year = []
El_cost_average_day = []

for i in range(365):
    for k in Electricity_price_read[i][0:24]:
        El_cost_year.append((k/1000)*1.11) #Prices in Euro/kWh, by dividing by 1000, times 1.1 to get 2022 euro values
          
    El_cost_average_day.append(((Electricity_price_read[i][24])/1000)*1.1)  #Prices in Euro/kWh that is why we are dividing by 1000, times 1.1 to get 2022 values


# --------------------Read load data for each hour both heating and electrical load---------
Load_data_read = pd.read_csv("Load_data_electricit_heating_2017.csv", header=0) #Takes values from January
Electricity_load_pd = Load_data_read["Electricty [kW]"]
Heating_load_pd = Load_data_read["Heating demand [kW]"]

Heating_load = np.zeros(8760) #in kWh
Electricity_load = np.zeros(8760) #in kWh

for count, i in enumerate(Heating_load_pd):
    Heating_load[count] = i

for count, i in enumerate(Electricity_load_pd):
    Electricity_load[count] = i


# -----------------------For Schedule inputs-------------------------

Energy_hourly_cost = np.array(El_cost_year)     #Prices in Euro/kWh
Average_median_cost_day = np.array(El_cost_average_day) 
Electricity_hourly_use = np.array(Electricity_load)
Heating_hourly_use = np.array(Heating_load)
ESS_charge_eff = 0.9
ESS_discharge_eff = 0.9

# Important to note that the maximum SoC for the battery is calculated in the schedule function

# ------------------------For NPV/LCOE inputs --------------------------
Lifetime_battery, Lifetime_project, Lifetime_ELH = 10, 10, 15  # in years
ESS_capacity_cost = 389.2   # in Euro(2022) per kWh (CAPEX) all cost included
ESS_power_cost = 148.8  # in Euro(2022) per kW (all cost included)
Fixed_ESS_O_and_M_cost = 4.19  # in Euro(2022) per kW-year
Variable_ESS_O_and_M_cost = 0.488/1000 # in Euro(2022) per kWh-year 
Discount_rate = 0.08 #8 percent
ELH_power_cost = 331.2  # In Euro(2022) per kW "UKSupplyCurve" #Lifetime said to be 15 years
ELH_OPEX = 1.5 # In euro(2022) per kW-year
Gas_cost = (30/1.1218)/293.07106944 #Euro/kWh   #1 mmBTU = 293.07106944 kWh, Gas cost 30 usd per million btu, change to euro: 1 euro is 1.1218 USD, 
                    #https://www.iea.org/data-and-statistics/charts/natural-gas-prices-in-europe-asia-and-the-united-states-jan-2020-february-2022
Peak_cost = 5.92/1.1218 #5.92 dollar (2022) per kW (max per month) change to euro: 1 euro is 1.1218 USD january 1 2022

# ------------- Add the heating and electricity together as the maximum possible ESSpower
total_energy_max=[]
for count, i in enumerate(Electricity_hourly_use):
    total_energy_max.append(
        Electricity_hourly_use[count] + Heating_hourly_use[count])


Case_3_data_LCOS = pd.read_csv('Results\Pygad_case_3_ESS_LCOS\ESS_power_LCOS_etc\Pygad_case_3_ESS_LCOS_200_gen.csv') #GA
Case_3_FF_data_LCOS = pd.read_csv('Results\Firefly_case_3_ESS_LCOS\ESS_power_LCOS_etc\Firefly_case_3_ESS_LCOS_200_gen.csv') #FF


Case_3_data_LCOS.sort_values('fitness_function', ascending=False, inplace=True, ignore_index=True)
Case_3_FF_data_LCOS.sort_values('fitness_function', ascending=False, inplace=True, ignore_index=True)

Solution_case_3_data_LCOS_FF = [Case_3_FF_data_LCOS["ESS_capacity"][0], Case_3_FF_data_LCOS["ESS_power"][0], Case_3_FF_data_LCOS["ELH_power"][0]] #capacity, power, ELH power
Solution_case_3_data_LCOS_GA = [Case_3_data_LCOS["ESS_capacity"][0], Case_3_data_LCOS["ESS_power"][0], Case_3_data_LCOS["ELH_power"][0]] #capacity, power, ELH power

Case3_ALL = [Solution_case_3_data_LCOS_GA, Solution_case_3_data_LCOS_FF]

Result_10_tries = [[],[],[],[],[],[],[],[],[],[],[]]
for i in Case3_ALL:
    ESS_capacity, ESS_power, ELH_power = i[0], i[1], i[2]
    
    Result_LCOS = fitness_func_LCOS(i)
    cost_investment=-((ESS_capacity_cost*ESS_capacity) + (ESS_power_cost*ESS_power)) + (ELH_power_cost*ELH_power)  # Adds the investment cost

    Result_10_tries[0].append(ESS_capacity)  # Capacity
    Result_10_tries[1].append(ESS_power)  # Power
    Result_10_tries[2].append(ELH_power) #ELH power
    Result_10_tries[3].append(Result_LCOS)# fittness function negative when looking for LCOS in GA
    Result_10_tries[4].append(cashflow_divided[0])  # cost charging
    Result_10_tries[5].append(cashflow_divided[1])  # Fixed OnM
    Result_10_tries[6].append(cashflow_divided[2])  # Variable OnM
    Result_10_tries[7].append(cashflow_divided[3])  # Cost yearly (summed OnM and charge cost)
    Result_10_tries[8].append(cost_investment)  # cost investment
    Result_10_tries[9].append(Schedule_sum[0])  # Charge energy to BESS for 1 year
    Result_10_tries[10].append(np.sum(Energy_yearly)) #Discharge energy from BESS total


raw_data = {'Algorithm':["GA","FF"],
    'ESS_power': Result_10_tries[1],
            'ESS_capacity': Result_10_tries[0],
            'ELH_power': Result_10_tries[2],
            'fitness_function': Result_10_tries[3],
            'cost_charge': Result_10_tries[4],
            'cost_O_n_M_fixed': Result_10_tries[5],
            'cost_O_n_m_variable': Result_10_tries[6],
            'Cost_total':Result_10_tries[7],
            'Cost_investment': Result_10_tries[8],
            'Summed_charge_kWh': Result_10_tries[9],
            'Summed_Discharge_kWh': Result_10_tries[10]}

df = pd.DataFrame(raw_data, columns = ['Algorithm', 'ESS_power', 'ESS_capacity', 'ELH_power', 'fitness_function',
                                        'cost_charge', 'cost_O_n_M_fixed', 'cost_O_n_m_variable', 'Cost_total',
                                        'Cost_investment', 'Summed_charge_kWh', 'Summed_Discharge_kWh'])

df.to_csv("Results\Divided_cost_results_200_gen\Case_3_LCOS_Divided_200_gen.csv", index=False, )