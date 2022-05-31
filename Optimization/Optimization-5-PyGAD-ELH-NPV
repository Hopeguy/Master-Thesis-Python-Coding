
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


def fitness_func_NPV(solution, solutionidx):
    """
    Returns the NPV value (used as fitness value in GA)
    """

    global Electricity_heating_load
    global cashflow_divided
    global ESS_capacity, ESS_power
    global Schedule_sum
    global Schedule
    global Schedule_capacity
    global ESS_capacity_year
    global Energy_hourly_use

    ESS_capacity, ESS_power, ELH_power = solution[0], solution[1], solution[2]

    # First we calculate the new electricity load and heating demand when an ELH is installed.
    # array first is the electricity load from ELH, and second array is the new Heating load
    Electricity_heating_load = Electricity_heater_load(
        Power=ELH_power, Heating_load=Heating_hourly_use)
    # adds the electricity load with the ELH electricity load
    Energy_hourly_use = Electricity_load + Electricity_heating_load[0]

    # First Year is just capex cost and negative as it is a cost
    cashflow_each_year = [-((ESS_capacity_cost*ESS_capacity) +
                            (ESS_power*ESS_power_cost) + ELH_power*ELH_power_cost)]
    ESS_capacity_year = 0  # Starts with zero energy in the storage

    cashflow_each_year = np.zeros(11)
    # First Year is just capex cost and negative as it is a cost
    cashflow_each_year[0] = -((ESS_capacity_cost *
                              ESS_capacity) + (ESS_power*ESS_power_cost))
    cashflow_divided = np.zeros(10)
    Schedule_sum = np.zeros(2)
    ESS_capacity_year = 0  # Starts with zero energy in the storage
    # starts at year 1 and includes year 10 as the lifetime of ESS is 10 years (battery)
    for year in range(10):

        Schedule = fun.ESS_schedule(ESS_capacity_size=ESS_capacity, ESS_power=ESS_power,
                                        Energy_hourly_cost=Energy_hourly_cost,
                                        Average_median_cost_day=Average_median_cost_day,
                                        Energy_hourly_use=Energy_hourly_use,
                                        ESS_discharge_eff=ESS_discharge_eff, ESS_charge_eff=ESS_charge_eff, Year=year, ESS_capacity_prev_year=ESS_capacity_year)

        # This calculates the cost of buying and using the ESS storage, as well as the profits of sell energy from it, and inputs that into an array for each year.
        # This does not include the energy used by the user. (Aka the load demand), but the schedule is designed from that schedule
        New_schedule = Schedule[0]
        # Inputs the preveious years ess capacity to next years
        ESS_capacity_year = Schedule[1]
        Schedule_capacity = Schedule[2]
        Peak_diff = fun.Peak_diff(
            Electricty_usage_pre_schedule=Energy_hourly_use, Schedule=Schedule[0])
        Cashflow_yearly = fun.cashflow_yearly_NPV(schedule_load=New_schedule[:, 0], schedule_discharge=New_schedule[:, 1], demand_cost=Energy_hourly_cost,
                                                        Variable_O_and_M_cost=Variable_ESS_O_and_M_cost, Fixed_O_and_M_cost=Fixed_ESS_O_and_M_cost,
                                                        ESS_power=ESS_power,  ELH_OPEX=ELH_OPEX, ELH_power=ELH_power, Gas_cost=Gas_cost,
                                                        Heating_demand_after_ELH=Electricity_heating_load[1], Heating_demand_pre=Heating_load,
                                                         Peak_diff=Peak_diff, Peak_diff_cost=Peak_cost, electricity_load_ELH=Electricity_heating_load[0])

        # print(Cashflow_yearly[1])
        for count, i in enumerate(Cashflow_yearly[1]):
            cashflow_divided[count] += i
        # Charge shcedule summed up kWh
        Schedule_sum[0]=np.sum(New_schedule[:, 0])
        # Discharge summed up in kWh
        Schedule_sum[1]=np.sum(New_schedule[:, 1])
        
        cashflow_each_year[year+1]=Cashflow_yearly[0]
        
        if year == 9:
            
            Residual_value = abs(fun.Residual_value_ELH(Interest_rate= Discount_rate, ELH_power_cost=ELH_power_cost, ELH_power=ELH_power, Lifetime_ELH = 15, project_lifetime=10))
            cashflow_each_year[year+1] += Residual_value
        
    
    
    fitness=fun.Fitness_NPV(discount_rate = Discount_rate,
                            cashflows = cashflow_each_year)

    return fitness  # positive as GA wants to maximize the NPV

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



### ----------------------Preparing other varuables-----------------------

fitness_function = fitness_func_NPV  #CHANGE BETWEEN LCOS OR NPV AS FITNESS FUNCTION

sol_per_pop = 50         #Number of solutions per population, Comparable to "agents in FF"
num_generations = 200     #number of generation to run the algorithm
num_parents_mating = int(sol_per_pop/2)     #number of solutions that will be mating (10% of total solutions used each generation)
init_range_low = 0.1                        #lowest value starting solutions can take
init_range_high = 2000                      #highest value starting solutions can take

parent_selection_type = "rank"               #Method choice for how to pick parent, can be: [sss, rws, sus, rank, random, tournament]
keep_parents = -1                           #Keeps all parents into the next generation (this is in order to not forget good solutions)

crossover_type = "uniform"      #method to crossover the genetics between the two parents, can be [singel_point, two_points, uniform, scattered, ]
crossover_probability = 0.8     #How likely it is for a parent to do a crossover, 0.8 is equal to 80%

mutation_type = "random"        #what operation the mutation will take, can be [random, swap, adaptive]
mutation_probability=0.1        # 10 percent chance of mutation operation to happen for a solution
gene_space = [{'low': 0.1, 'high': 8000}, {'low': 0.1, 'high': np.max(Electricity_load)}, {'low': 0.1, 'high':np.max(Heating_hourly_use)}]

#The gene_space is the search space for the capacity, ess_power and ELH_power, where the maximum is the max value found in the array as it will never have to be higher than that.

#It should be implemented that the maximum power the solution can take should be dependent on the max value from the load data.



#------------------------Gets the average cost for each day, and the hourly cost at each hour during the year------------------------

# -----------set up FF algorithm----------
Result_10_tries = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

for i in range(10):

     #-----------Set up ga-------------
    ga_instance = pygad.GA(num_generations=num_generations,
                       allow_duplicate_genes= False,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=3,
                       gene_type = float,
                       #init_range_low=init_range_low,
                       #init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       #callback_generation=callback_gen,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_probability = mutation_probability,
                       gene_space=gene_space,
                       #stop_criteria = "saturate_7",    #Stop the algorithm if the same fitness value is given for 7 consectuive generations
                       save_solutions=True,
                       save_best_solutions= True,
                       suppress_warnings=True) 
    start = time.time() 
    ga_instance.run()
    end = time.time()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    cost_investment=-((ESS_capacity_cost*solution[0]) + (ESS_power_cost*solution[1])) + (ELH_power_cost*solution[2])  # Adds the investment cost

    Result_10_tries[0].append(solution[0])  # Capacity
    Result_10_tries[1].append(solution[1])  # Power
    Result_10_tries[2].append(solution[2]) #ELH power
    Result_10_tries[3].append(solution_fitness)# fittness function positive when looking for NPV
    Result_10_tries[4].append(abs(end-start))  # Times
    Result_10_tries[5].append(cashflow_divided[0])  # profit from selling kWh
    Result_10_tries[6].append(cashflow_divided[1])  # profit from Peak_kW
    Result_10_tries[7].append(cashflow_divided[2])  # Profit for saved heating total (saved minus cost of electricty for heating)
    Result_10_tries[8].append(cashflow_divided[3]) # saved money on using electrical heating instead of gas
    Result_10_tries[9].append(cashflow_divided[4]) # Cost of using electrical heating
    Result_10_tries[10].append(cashflow_divided[5])  # cost from chargin
    Result_10_tries[11].append(cashflow_divided[6])  # cost OnM fixed
    Result_10_tries[12].append(cashflow_divided[7])  # Cost OnM Variable
    Result_10_tries[13].append(cashflow_divided[8])  # OPEX ELH
    Result_10_tries[14].append(cashflow_divided[9])  # Cashflow_total
    Result_10_tries[15].append(cost_investment)  # cost investment
    Result_10_tries[16].append(Schedule_sum[0])  # Charge energy to BESS
    Result_10_tries[17].append(Schedule_sum[1])  # Discharge energy from BESS

    print(Result_10_tries[3])




Schedule = fun.ESS_schedule(ESS_capacity_size = (solution[0]), ESS_power =(solution[1]),
                                        Energy_hourly_cost = Energy_hourly_cost,
                                        Average_median_cost_day = Average_median_cost_day,
                                        Energy_hourly_use = Energy_hourly_use,
                                        ESS_discharge_eff = ESS_discharge_eff,
                                        ESS_charge_eff = ESS_charge_eff, Year = 10,
                                        ESS_capacity_prev_year = ESS_capacity_year)

Average_hourly_cost=[]
for count, average in enumerate(Average_median_cost_day):
    for i in range(24):
        Average_hourly_cost.append(average)


Charge_discharge=[]

for count, hour_cost in enumerate(Energy_hourly_cost):
    if hour_cost < Average_hourly_cost[count]:
        Charge_discharge.append('Charge+++++')
    else:
        Charge_discharge.append('Discharge-----')

Schedule_last_year = [[], [], [], [],[],[],[], [], []]
Schedule_last_year[0]= Schedule[0][:, 0]  # Charge schedule (What is taken from grid)
Schedule_last_year[1]= Schedule[0][:, 1]  # Discharge schedule (What is given to the using user, More is taken from the capacity storage bc of efficency losses)
Schedule_last_year[2]=Schedule[2]  # Capacity schedule
Schedule_last_year[3]= Energy_hourly_use  # Total Electricity
Schedule_last_year[4]=np.array(Average_hourly_cost)  # Averagey hourly cost for electricy each day
Schedule_last_year[5]=np.array(El_cost_year)  # Cost each hour for electricity
Schedule_last_year[6]=Charge_discharge  # if we want to charge or discharge at this hour
Schedule_last_year[7]=Electricity_heating_load[1]  # Remaining gas heating left
Schedule_last_year[8]=Heating_hourly_use  # gas heating before ELH

data_schedule={'charge': Schedule_last_year[0],
                'discharge': Schedule_last_year[1],
                'capacity': Schedule_last_year[2],
                'Energy_use': Schedule_last_year[3],
                'Average_daily_cost': Schedule_last_year[4],
                'Energy_hourly_cost':Schedule_last_year[5],
                'Charge_or_discharge': Schedule_last_year[6],
                'Gas_heating_needed': Schedule_last_year[7],
                'Gas_heating_before_ELH_installment': Schedule_last_year[8]}



tf = pd.DataFrame(data_schedule, columns=['charge', 'discharge', 'capacity',
                             'Energy_use', 'Average_daily_cost', 'Energy_hourly_cost',
                              'Charge_or_discharge', 'Gas_heating_needed', 'Gas_heating_before_ELH_installment'])


std_ESS_power = np.std(Result_10_tries[1]) 
std_ESS_capacity = np.std(Result_10_tries[0])
std_fitness_function = np.std(Result_10_tries[3])
std_ELH_power = np.std(Result_10_tries[2])


raw_data = {'ESS_power': Result_10_tries[1],
            'ESS_capacity': Result_10_tries[0],
            'ELH_power': Result_10_tries[2],
            'fitness_function': Result_10_tries[3],
            'Time': Result_10_tries[4],
            'std_ESS_power': std_ESS_power,
            'std_ESS_capacity': std_ESS_capacity,
            'std_fitness_function': std_fitness_function,
            'std_ELH_power': std_ELH_power,
            'profit_kWh': Result_10_tries[5],
            'profit_peak_kW': Result_10_tries[6],
            'profit_saved_gas_heating_total': Result_10_tries[7],
            'Saved_cost_heating': Result_10_tries[8],
            'heating_cost_electricity': Result_10_tries[9],
            'cost_charge': Result_10_tries[10],
            'cost_O_n_M_fixed': Result_10_tries[11],
            'cost_O_n_m_variable': Result_10_tries[12],
            'OPEX_ELH': Result_10_tries[13],
            'Cashflow_total': Result_10_tries[14],
            'Cost_investment': Result_10_tries[15],
            'Summed_charge_kWh': Result_10_tries[16],
            'Summed_Discharge_kWh': Result_10_tries[17]}

df = pd.DataFrame(raw_data, columns = ['ESS_power', 'ESS_capacity', 'ELH_power', 'fitness_function',
                                           'Time', 'std_ESS_power', 'std_ESS_capacity', 'std_fitness_function',
                                           'std_ELH_power', 'profit_kWh', 'profit_peak_kW',
                                            'profit_saved_gas_heating_total', 'Saved_cost_heating', 'heating_cost_electricity',
                                           'cost_charge', 'cost_O_n_M_fixed', 'cost_O_n_m_variable', 'OPEX_ELH', 'Cashflow_total',
                                           'Cost_investment', 'Summed_charge_kWh', 'Summed_Discharge_kWh'])

save_file_name_std = f"Results\Pygad_case_3_ESS_NPV\ESS_power_NPV_etc\Pygad_case_3_ESS_NPV_{num_generations}_gen.csv"

save_file_name_schedule = f"Results\Pygad_case_3_ESS_NPV\Charge_discharge_capacity\Pygad_case_3_ESS_NPV_{num_generations}_gen_Sch_year_10.csv"

df.to_csv(save_file_name_std, index=False, )
tf.to_csv(save_file_name_schedule, index=False, )
    

save_file_name_fittnes_plot = f"Results\\Pictures_etc\\Pygad-case-3-NPV-convergence\\fitness_NPV_over_generation_{num_generations}_gen.jpeg"
save_file_name_genes_plot = f"Results\\Pictures_etc\\Pygad-case-3-NPV-convergence\\Best_genes_NPV_{num_generations}_gen.jpeg"
save_file_name_solution_rate_plot = f"Results\\Pictures_etc\\Pygad-case-3-NPV-convergence\\New_Solution_rate_NPV_{num_generations}_gen.jpeg"

ga_instance.plot_fitness(title= "GA Case 3: NPV", xlabel= "Generation",
                ylabel="NPV [Euro]", plot_type="plot", save_dir=save_file_name_fittnes_plot)
ga_instance.plot_genes(graph_type='plot', title= "GA case 3: Best Genes NPV", solutions="best", save_dir=save_file_name_genes_plot)
ga_instance.plot_new_solution_rate(title = "Ga Case 3: New solution rate vs Generation NPV", ylabel= "New solutions explored", save_dir = save_file_name_solution_rate_plot)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(solution, solution_fitness)