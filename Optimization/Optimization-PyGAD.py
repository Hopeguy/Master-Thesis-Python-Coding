# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:46:11 2022

@author: jocke
"""

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
import numpy_financial as npf
import functions as fun
import time
import pygad


def fitness_func(solution, solution_idx):
    discount_rate = 0.08
    

    ESS_capacity, ESS_power = solution[0], solution[1]/10
    cashflow_each_year = [-((ESS_capacity_cost*ESS_capacity) + (ESS_power*ESS_power_cost))] #First Year is just capex cost and negative as it is a cost

    for year in range(1,11): # starts at year 1 and includes year 10
                
        Schedule = fun.ESS_schedule(ESS_capacity_size=ESS_capacity, ESS_power=ESS_power,
                                        Energy_hourly_cost=Energy_hourly_cost,
                                        Average_median_cost_day=Average_median_cost_day,
                                        Energy_hourly_use=Energy_hourly_use,
                                        ESS_discharge_eff=ESS_discharge_eff, ESS_charge_eff=ESS_charge_eff, Year = year)
    
                
        #This calculates the cost of buying and using the ESS storage, as well as the profits of sell energy from it, and inputs that into an array for each year.
        #This does not include the energy used by the user. (Aka the load demand), but the schedule is designed from that schedule
        cashflow_each_year.append(fun.cashflow_yearly(schedule_load = Schedule[:, 0], schedule_discharge = Schedule[:,1], demand_cost = Energy_hourly_cost))

    fitness = fun.Fitness_NPV(discount_rate = discount_rate, cashflows = cashflow_each_year)
    return fitness

def callback_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])

### --------Preparing other varuables---------

num_generations = 5
num_parents_mating = 2
fitness_function = fitness_func
sol_per_pop = 10 #Number of solutions per population
init_range_low = 0
init_range_high = 100

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 5

gene_space = np.array([range(0, 1000), range(0, 100)]) 

Battery_size = list(range(1, 101))  # kWh
Battery_power = list(range(1, 101))  # kW
functions_inputs = np.array(Battery_power)

#---------------------------------------------

#-----------Set up ga-------------
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=2,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       callback_generation=callback_gen,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       gene_space=gene_space,
                       gene_type = int)



### Input values for solution matrix ###


# ----------Gets the average cost for each day, and the hourly cost at each hour during the year--------

Electricity_price_read_oslo = np.genfromtxt(
    "os-eur17.csv", delimiter=",")  # Prices in EUR/MWh
El_cost_year = []
El_cost_average_day = []
for i in range(365):
    for k in Electricity_price_read_oslo[i][0:24]:
        El_cost_year.append(k)
    El_cost_average_day.append(Electricity_price_read_oslo[i][24])

# -------------Read load data for each hour of a year of house 59---------
El_data_read = pd.read_csv("home59_hall687_sensor1506c1508_electric-mains_electric-combined.csv",
                           header=None)

# Average W used for that hour (giving the Wh) from second of january
El_data_59 = El_data_read[1][2038:8760+2038]

power_load_59 = []
for i in El_data_59:
    power_load_59.append(i/1000)  # in kWh

power_load_59 = np.array(power_load_59)
# --------------------------------------------------------------------------


# ---------------Input values (non changable)---------------------------------------'
# ------For Schedule inputs-----------

Energy_hourly_cost = np.array(El_cost_year)
Average_median_cost_day = np.array(El_cost_average_day)
Energy_hourly_use = power_load_59
ESS_charge_eff = 0.9
ESS_discharge_eff = 0.9


# For schedule the matrix containing the batterys different power/capacity:

Battery_size = list(range(0, 101))  # kWh
Battery_power = list(range(0, 101))  # kW

# Important to note that the maximum SoC for the battery is calculated in the schdule function
# Only import is the Total max size that is also used for calculating the cost

# ------For NPV/max inputs -------------
Lifetime_battery = 10  # in years
interest_rate = 0.07  # 7 percent
ESS_capacity_cost = 10  # in pound per kWh
ESS_power_cost = 2  # in pound per kW
ESS_O_and_M_cost = 1  # in Pound per kWh
Discount_rate = 0.08 #8 percent


start = time.time()
ga_instance.run()

end = time.time()

ga_instance.plot_fitness()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

print(abs(start-end))