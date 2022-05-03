import pandas as pd
import numpy as np
import numpy_financial as npf
import functions as fun
import time
import pygad

def fitness_func_NPV(solution, solution_idx):
    """
    Returns the NPV value (used as fitness value in GA)
    """

    ESS_capacity, ESS_power = solution[0], solution[1]
    
    cashflow_each_year = [-((ESS_capacity_cost*ESS_capacity) + (ESS_power*ESS_power_cost))] #First Year is just capex cost and negative as it is a cost
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
        cashflow_each_year.append(fun.cashflow_yearly_NPV(schedule_load = New_schedule[:, 0], schedule_discharge = New_schedule[:,1], demand_cost = Energy_hourly_cost,
                                                          Variable_O_and_M_cost = Variable_ESS_O_and_M_cost, Fixed_O_and_M_cost = Fixed_ESS_O_and_M_cost, ESS_power = ESS_power))

    fitness_NPV = fun.Fitness_NPV(discount_rate = Discount_rate, cashflows = cashflow_each_year)
    #print("fittnes =", fitness, "power, Capacity = ", ESS_power, ESS_capacity)
    return fitness_NPV


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

### --------Preparing other varuables---------


fitness_function = fitness_func_NPV  #CHANGE BETWEEN LCOS OR NPV AS FITNESS FUNCTION

sol_per_pop = 5           #Number of solutions per population, Comparable to "agents in FF"
num_generations = 5     #number of generation to run the algorithm
num_parents_mating = int(sol_per_pop/2)     #number of solutions that will be mating (10% of total solutions used each generation)
init_range_low = 1          #lowest value starting solutions can take
init_range_high = 10      #highest value starting solutions can take

parent_selection_type = "rank"      #Method choice for how to pick parent, can be: [sss, rws, sus, rank, random, tournament]
keep_parents = -1       #Keeps all parents into the next generation (this is in order to not forget good solutions)

crossover_type = "uniform"      #method to crossover the genetics between the two parents, can be [singe_point, two_points, uniform, scattered, ]
crossover_probability = 0.8     #How likely it is for a parent to do a crossover, 0.8 is equal to 80%

mutation_type = "random"        #what operation the mutation will take, can be [random, swap, adaptive]
mutation_probability=0.1        # 10 percent chance of mutation operation to happen for a solution
gene_space = {'low': 1, 'high': 2000}#np.array([range(1, 2000), range(1, 2000)]) 


#---------------------------------------------

#-----------Set up ga-------------
ga_instance = pygad.GA(num_generations=num_generations,
                       allow_duplicate_genes= True,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=2,
                       gene_type = int,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       #callback_generation=callback_gen,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_probability = mutation_probability,
                       gene_space=gene_space,
                       #stop_criteria = "saturate_7",    #Stop the algorithm if the same fitness value is given for 7 consectuive generations
                       save_solutions=True) 



### Input values for solution matrix ###


# ----------Gets the average cost for each day, and the hourly cost at each hour during the year--------

Electricity_price_read = np.genfromtxt(
    "sto-eur17.csv", delimiter=",")  # Prices in EUR/MWh
El_cost_year = []
El_cost_average_day = []

for i in range(365):
    for k in Electricity_price_read[i][0:24]:
        El_cost_year.append((k/1000)*1.11) #Prices in Euro/Kwh, by dividing by 1000, times 1.1 to get 2022 euro values
          
    El_cost_average_day.append(((Electricity_price_read[i][24])/1000)*1.1)  #Prices in Euro/kwh that is why we are dividing by 1000, times 1.1 to get 2022 values


# -------------Read load data for each hour of a year of house 59---------
El_data_read = pd.read_csv("home59_hall687_sensor1506c1508_electric-mains_electric-combined.csv",
                           header=None)

# Average W used for that hour (giving the Wh) from second of january
El_data_59 = El_data_read[1][2038:8760+2038]

power_load_59 = []
for i in El_data_59:
    power_load_59.append((i/1000))  # in kWh devide with 1000 to get it in kWh

# --------------------------------------------------------------------------


# ---------------Input values (non changable)---------------------------------------'
# ------For Schedule inputs-----------

Energy_hourly_cost = np.array(El_cost_year)
Average_median_cost_day = np.array(El_cost_average_day)
Energy_hourly_use = np.array(power_load_59)
ESS_charge_eff = 0.9
ESS_discharge_eff = 0.9


# Important to note that the maximum SoC for the battery is calculated in the schdule function
# Only import is the Total max size that is also used for calculating the cost


# ------For NPV/LCOE inputs -------------
Lifetime_battery = 10  # in years
ESS_capacity_cost = 427.31   # in Euro(2022) per kWh (CAPEX) all cost included
ESS_power_cost = 1710.2  # in Euro(2022) per kW (all cost included)
Fixed_ESS_O_and_M_cost = 4.19  # in Euro(2022) per kWh-year
Variable_ESS_O_and_M_cost = 0.488/1000 # in Euro(2022) per kWh-year 
Discount_rate = 0.08 #8 percent


start = time.time()
ga_instance.run()

end = time.time()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
if fitness_function == fitness_func_LCOS:
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness = -solution_fitness))
elif fitness_function == fitness_func_NPV:
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness = solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

print(abs(start-end))

ga_instance.plot_fitness()
ga_instance.plot_genes()