
import pandas as pd
import numpy as np
import numpy_financial as npf
import functions as fun
import time
import pygad
import matplotlib.pyplot as plt

def fitness_func_NPV(solution, solution_idx):
    """
    Returns the NPV value (used as fitness value in GA)
    """
    global cashflow_divided
    global ESS_capacity, ESS_power
    ESS_capacity, ESS_power = solution[0], solution[1]
    
    cashflow_each_year = [-((ESS_capacity_cost*ESS_capacity) + (ESS_power*ESS_power_cost))] #First Year is just capex cost and negative as it is a cost
    cashflow_divided = np.zeros(7)
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
        Peak_diff = fun.Peak_diff(Electricty_usage_pre_schedule = Energy_hourly_use, Schedule = Schedule[0])
        Cashflow_yearly = fun.cashflow_yearly_NPV(schedule_load = New_schedule[:, 0], schedule_discharge = New_schedule[:,1], demand_cost = Energy_hourly_cost,
                                                        Variable_O_and_M_cost = Variable_ESS_O_and_M_cost, Fixed_O_and_M_cost = Fixed_ESS_O_and_M_cost,
                                                        ESS_power = ESS_power, Peak_diff = Peak_diff, Peak_diff_cost = Peak_cost)
        #print(Cashflow_yearly[1])
        for count, i in enumerate(Cashflow_yearly[1]):
            cashflow_divided[count] += i

        cashflow_each_year.append(Cashflow_yearly[0])
    fitness = fun.Fitness_NPV(discount_rate = Discount_rate, cashflows = cashflow_each_year)
    return fitness #negative as the FF algo want to minimize the fitness function



def fitness_func_LCOS(solution, solution_idx):
    
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
                                        ESS_discharge_eff=ESS_discharge_eff, ESS_charge_eff=ESS_charge_eff, Year = year, ESS_capacity_prev_year = ESS_capacity_year)
    
                
        #This calculates the cost of buying and using the ESS storage, as well as the profits of sell energy from it, and inputs that into an array for each year.
        #This does not include the energy used by the user. (Aka the load demand), but the schedule is designed from that schedule
        New_schedule = Schedule[0]
        ESS_capacity_year = Schedule[1]  #Inputs the preveious years ess capacity to next years
        Cost_yearly.append(fun.Cost_yearly_LCOS(schedule_load = New_schedule[:,0], schedule_discharge = New_schedule[:,1], demand_cost = Energy_hourly_cost,
                             Fixed_O_and_M_cost = Fixed_ESS_O_and_M_cost, Variable_O_and_M_cost = Variable_ESS_O_and_M_cost, ESS_power = ESS_power))
        
        Energy_yearly.append(np.sum(New_schedule[:,1]))

    fitness_LCOS = fun.Fittnes_LCOS(discount_rate = Discount_rate, CAPEX = CAPEX, Yearly_cost = Cost_yearly, Yearly_energy_out = Energy_yearly)
    return -fitness_LCOS #negative as the function want to maximize but we want the lowest value for LCOS in Euro / kWh


def callback_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])
    #print("population: ", ga_instance.population)


Electricity_price_read = np.genfromtxt(
    "os-eur17.csv", delimiter=",")  # Prices in EUR/MWh
El_cost_year = []
El_cost_average_day = []

for i in range(365):
    for k in Electricity_price_read[i][0:24]:
        El_cost_year.append((k/1000)*1.11) #Prices in Euro/kWh, by dividing by 1000, times 1.1 to get 2022 euro values
          
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
ESS_capacity_cost = 427.31   # in Euro(2022) per kWh (CAPEX) all cost included
ESS_power_cost = 1710.2  # in Euro(2022) per kW (all cost included)
Fixed_ESS_O_and_M_cost = 4.19  # in Euro(2022) per kW-year
Variable_ESS_O_and_M_cost = 0.488/1000 # in Euro(2022) per kWh-year 
Discount_rate = 0.08 #8 percent   
Peak_cost = 5.92/1.1218 #5.92 dollar (2022) per kW (max per month) change to euro: 1 euro is 1.1218 USD january 1 2022


### --------Preparing other varuables---------

fitness_function = fitness_func_NPV  #CHANGE BETWEEN LCOS OR NPV AS FITNESS FUNCTION

sol_per_pop = 10          #Number of solutions per population, Comparable to "agents in FF"
num_generations = 100     #number of generation to run the algorithm
num_parents_mating = int(sol_per_pop/2)     #number of solutions that will be mating (50% of total solutions used each generation)
init_range_low = 0.1          #lowest value starting solutions can take
init_range_high = 2000     #highest value starting solutions can take

parent_selection_type = "rank"      #Method choice for how to pick parent, can be: [sss, rws, sus, rank, random, tournament]
keep_parents = -1       #Keeps all parents into the next generation (this is in order to not forget good solutions)

crossover_type = "uniform"      #method to crossover the genetics between the two parents, can be [singe_point, two_points, uniform, scattered, ]
crossover_probability = 0.9     #How likely it is for a parent to do a crossover, 0.8 is equal to 80%

mutation_type = "random"        #what operation the mutation will take, can be [random, swap, adaptive]
mutation_probability=0.1        # 10 percent chance of mutation operation to happen for a solution
gene_space = [{'low': 0.1, 'high': 2000}, {'low': 0.1, 'high': np.max(Electricity_load)}]

#It should be implemented that the maximum power the solution can take should be dependent on the max value from the load data.

#---------------------------------------------





# ----------Gets the average cost for each day, and the hourly cost at each hour during the year--------

Result_10_tries = [[],[],[]]

for i in range(10):
    #-----------Set up ga-------------
    ga_instance = pygad.GA(num_generations=num_generations,
                       allow_duplicate_genes= True,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=2,
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
                       save_solutions=True) 
    start = time.time() 
    ga_instance.run()
    end = time.time()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    Result_10_tries[0].append(solution)
    Result_10_tries[1].append(solution_fitness)
    Result_10_tries[2].append(abs(end-start))
    print(Result_10_tries[1])



solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
if fitness_function == fitness_func_LCOS:
    print("Fitness value of the best solution NPV = {solution_fitness}".format(solution_fitness = -solution_fitness), "Euro/kWh")
elif fitness_function == fitness_func_NPV:
    print("Fitness value of the best solution LCOS = {solution_fitness}".format(solution_fitness = solution_fitness),"Euro")
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

"""print(abs(start-end))
#ga_instance.plot_fitness()
ga_instance.plot_genes(graph_type="histogram", solutions='all')

cost_investment = -((ESS_capacity_cost*ESS_capacity) + (ESS_power*ESS_power_cost))

cashflow_divided[6] = cost_investment  #Adds the investment cost
#cashflow_divided[7] = solution #Adds the NPV to the list of all the cost and profits
#Plot of all the cost and profits over the tne years divided
print("Profit kWh: ",cashflow_divided[0], "Profit peak: ", cashflow_divided[1])
print("cost charge: ", cashflow_divided[2], "cost OnM fixed: ", cashflow_divided[3], "cost OnM variable: ", cashflow_divided[4])
print("Total cashflow: ", cashflow_divided[5], "Cost investment: ", cost_investment)

#Plot of all the cost and profits over the tne years divided
print(cashflow_divided)
fig, ax = plt.subplots()
ax.bar(["Profit_kWh", "Profit_peak", "cost_charge", "Cost_OnM_fixed", "cost_OnM_variable", "Total_cashflow","Investment cost: "], cashflow_divided, width=1, edgecolor="white", linewidth=1)

plt.show()"""

plt.plot(Result_10_tries[0])
plt.plot(Result_10_tries[1])