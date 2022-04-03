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


### Function for solving the ESS schedule ###




import pandas as pd
import numpy as np
import numpy_financial as npf
import functions as fun
import time


def Crossover_bit(Parent_1, Parent_2):
    """
    Parent 1 and 2 are the two solutions that are doing a cross over, and should be a
    list with both the binary capacity and power for the each parent. If you use elitism take the
    best parents together
    """
    # For single ESS (no hybrid)

    # We want to swap the power between the parents, one is enough to generate the two new
    #Solutions (offspring)
    for count, bit_power in enumerate(Parent_2[1]):
        if bit_power == 1:
            save = count
            Parent_2[1][count] = 0

    for count2, bit_power2 in enumerate(Parent_1[1]):
        if bit_power2 == 1:
            Parent_2[1][count2] = 1
            Parent_1[1][count2] = 0

    Parent_1[1][save] = 1

    Offspring_1 = Parent_1
    Offspring_2 = Parent_2

    return [Offspring_1, Offspring_2]


def Crossover(Parent_1, Parent_2):
    """
    We only enter this function if the criteria of crossover have been accepted in the 
    genetic algorithm for example, 70% chance.
    As we know that the value for the power and capacity is the same as the pos
    in the list we dont need to use bit here,the goal is to swap the power between both 
    parents and that will generate two new offsprings.
    Parent 1 and 2 is a list with 3 values, power, capacity and fitness value.
    we want to get an offspring that is only the capacity and power, so a list with 2 values
    """
    Offspring_1 = [Parent_1[0], Parent_2[1]]  #Swaps the power from the second parent to the first one
    Offspring_2 = [Parent_2[0], Parent_1[1]]    #Swaps the power from the first parent to the second one
    
    return [Offspring_1, Offspring_2]


# ------------To test the crossover function----------------

# parent_1 = [[0,1,0],[1,0,0]] #binary representation of a capacity and power for a solution
# parent_2 = [[1,0,0],[0,1,0]] #the first list is the capacity, the second is the power

# parent_1 = [5,8]
# parent_2 = [9,1]

# crossover = Crossover(Parent_1 = parent_1, Parent_2 = parent_2)

# print(crossover)

# ----------------------------------------------------------


# ---------------Mutation function--------------------

def mutation(Solution):  # if a mutaiton has happend the this function is called
    """
    bit_string is the binary string for that a mutation would happen on
    As we give the solution as an input, we take the value as the pos in the bit string
    so we get a bit_string that we can do the bitstring on.

    """
    bit_string = np.zeros(100, int)
    bit_string[Solution] = 1
    
    # Goes through the list of binary bits
    for pos, bit in enumerate(bit_string):
        if bit == 1:  # When we find the active bit we want to deactive it "== 0"
            # Checks so we dont accidently activated the same bit again with the random function
            while bit_string[pos] == 1:
                if bit_string[pos] == 1:  # Check that the bit is = 1 to swap it
                    bit_string[pos] = 0  # Swaps the bit to zero "Turn it off"
                    # Randomly starts another bit (putting it == 1)
                    bit_string[np.random.randint(0, len(bit_string))] = 1
            break  # When we have made a new bit that was not the old one activated we break the loop and return the new binary code list

    result = np.where(bit_string == 1)
    return result[0][0] # Returns the new binary code list where the mutation has happend, where it has swaped the bit to another spot

# -----------To test the mutaion function--------
# mutation_string = mutation(bit_string = [0, 1, 0])

# print(mutation_string)

# -----------------------------------------------



def Genetic_algorithm(Population_size, Mutation_rate, Crossover_rate, generations, Energy_hourly_cost,
                      Average_median_cost_day, Energy_hourly_use, ESS_discharge_eff, ESS_charge_eff,
                      ESS_capacity_cost, ESS_power_cost, ESS_O_and_M_cost, Base_case_cost, Discount_rate):
    
    # fitness_diff = 5000
    Generation_best_solution = []
    Battery_capacity = list(range(1, 101))  # kWh
    Battery_power = list(range(1, 101))  # kW

    Population = []    #Rework this later with np.arrays, and np.empty


    for i in range(Population_size): #This solution expects that the pos in the power and capacity is the same as its value

        Solution_power = np.random.randint(0, len(Battery_power))
        Solution_capacity = np.random.randint(0, len(Battery_capacity))
        Solution = [Solution_capacity, Solution_power]
        Population.append(Solution)
    



    for solution in Population:
        ESS_capacity, ESS_power = solution[0], solution[1]
        cashflow_each_year = [-((ESS_capacity_cost*ESS_capacity) + (ESS_power*ESS_power_cost))] #First Year is just capex cost and negative as it is a cost
        for year in range(1,11): #Starts at year 1 and includes year 10 aswell
            
            Schedule = fun.ESS_schedule(ESS_capacity_size=ESS_capacity, ESS_power=ESS_power,
                                    Energy_hourly_cost=Energy_hourly_cost,
                                    Average_median_cost_day=Average_median_cost_day,
                                    Energy_hourly_use=Energy_hourly_use,
                                    ESS_discharge_eff=ESS_discharge_eff, ESS_charge_eff=ESS_charge_eff, Year = year)

            
            #This calculates the cost of buying and using the ESS storage, as well as the profits of sell energy from it, and inputs that into an array for each year.
            #This does not include the energy used by the user. (Aka the load demand), but the schedule is designed from that schedule
            cashflow_each_year.append(fun.cashflow_yearly(schedule_load = Schedule[:, 0], schedule_discharge = Schedule[:,1], demand_cost = Energy_hourly_cost))
                                         
        solution.append(fun.Fitness_NPV(discount_rate = Discount_rate, cashflows = cashflow_each_year))
      
    solution = np.array(solution)
    
   
    Population.sort(key=lambda i: i[2], reverse=True)  # Sort the list dependent on the fitness value
    generation = 0
    while generation < generations:  # or fitness_diff < (10^-1):

        #First step is the parent selection:
        
        
        # first we do crossover if the crossover rate is fine, this we do for each combination of parents (elitims as we do 0-1 togehter,  2-3 togheter etc)
        counter = 0
        New_population = [] #new list to store the newly generated population with crossover offspring
        while counter < len(Population) - 1:
            if np.random.rand() < Crossover_rate: #Checks if the random value is lower than crossover rate, then we enter the function
                # print("crossover happend", Population[counter], Population[counter+1])
                
                Offspring = Crossover(Parent_1 = Population[counter], Parent_2 = Population[counter + 1])
                New_population.append(Offspring[0])
                New_population.append(Offspring[1])
                # print(Offspring[0], Offspring[1])
            else: #If the crossover critera is not met we just input the population values as the were before
                New_population.append(Population[counter][0:2])  #Deletes the fiteness value from the solution list, just takes power and capacity
                New_population.append(Population[counter+1][0:2])  #deletes the fitness value from the solution list

            counter += 2 #Jumps two steps as we always take the parents as pairs
        Population = New_population     #Replaces the previous population with the new crossover one. 
                
        
        #Secondly we try to apply the mutation function if the criteria is met
       
        New_population = []
        for solution in Population: #will go through the whole population
            if np.random.rand() < Mutation_rate: #first we look if we wnat to mutate the solution
                if np.random.rand() < 0.5:  #here we check if we should mutate the power or capacity of the solution
                    #First we change the capcity if this is true
                    New_population.append([mutation(Solution = solution[0]), solution [1]])
                    
                else: #Here we instead mutate the power
                    New_population.append([solution [0], mutation(Solution = solution[1])]) 
            else:
                New_population.append(solution) #If we dont have a mutation we just add the solution back 
                
        
        Population = New_population

        for count, solution in enumerate(Population):

            ESS_capacity, ESS_power = solution[0], solution[1]
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
                                             
            solution.append(fun.Fitness_NPV(discount_rate = Discount_rate, cashflows = cashflow_each_year))
        
        
        Population.sort(key=lambda i: i[2], reverse=True)
        # print(Population[0], generation, "-------------------------")
        Generation_best_solution.append(Population[0])
        generation += 1



    return Generation_best_solution # best_solution, generation


# ----------------------------------


### Input values for solution matrix ###

Battery_size = list(range(1, 101))  # kWh
Battery_power = list(range(1, 101))  # kW

for count, power in enumerate(Battery_power):
    Battery_power[count] = power/10


Battery_size_binary = np.zeros(100, dtype=int)
Battery_power_binary = np.zeros(100, dtype=int)


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
ESS_capacity_cost = 100  # in pound per kWh
ESS_power_cost = 20  # in pound per kW
ESS_O_and_M_cost = 1  # in Pound per kWh
Discount_rate = 0.07 #7


# ---------For Genetic inputs---------
Population_size = 5 #have to be even
Mutation_rate = 1/Population_size
Crossover_rate = 0.7
Generation = 10

### Calculate Base case cost ####
Base_case_cost = 0  # in  pence
for Count, El in enumerate(power_load_59):
    # Divide by 1000 to get how many kWh
    Base_case_cost += (El)*(El_cost_year[Count])
    
# -------------Test schedule function-------


# Schedule = ESS_schedule(ESS_capacity_size=1, ESS_power=0.1, Energy_hourly_cost=Energy_hourly_cost,
#                         Average_median_cost_day=Average_median_cost_day,
#                         Energy_hourly_use=Energy_hourly_use, ESS_charge_eff=ESS_charge_eff,
#                         ESS_discharge_eff=ESS_discharge_eff)

# print(Schedule)

# # positive as discharge are negative values
# New_load_demand = power_load_59 + Schedule[:, 0]
# New_case_cost = 0
# for Count_2, El_2 in enumerate(New_load_demand):
#     New_case_cost += (El_2/1000)*(El_cost_year[Count_2])


# Saving = Base_case_cost - New_case_cost  #The difference is the savings in money by installing ESS
# print(Saving)

# Saving_2 = Fitness_max_saved(Energy_hourly_use = Energy_hourly_use, schedule = Schedule, Energy_hourly_cost=Energy_hourly_cost,
#                              ESS_power = 0.1, ESS_capacity = 1, ESS_capacity_cost = ESS_capacity_cost,
#                              ESS_power_cost = ESS_power_cost, ESS_O_and_M_cost = ESS_O_and_M_cost,
#                              Base_case_cost = Base_case_cost)
# print(Saving_2)


#---------------RUNNING THE GENETIC ALGOTIHM
start = time.time()

gen_tries = Genetic_algorithm(Population_size=Population_size, Mutation_rate=Mutation_rate, #Returns a list of each generations best solution
                              Crossover_rate=Crossover_rate, generations=Generation,
                              Energy_hourly_cost=Energy_hourly_cost,
                              Average_median_cost_day=Average_median_cost_day,
                              Energy_hourly_use=Energy_hourly_use,
                              ESS_discharge_eff=ESS_discharge_eff,
                              ESS_charge_eff=ESS_charge_eff,
                              ESS_capacity_cost=ESS_capacity_cost,
                              ESS_power_cost=ESS_power_cost,
                              ESS_O_and_M_cost=ESS_O_and_M_cost,
                              Base_case_cost=Base_case_cost,
                              Discount_rate = Discount_rate)
end = time.time()

# gen_tries[1] = gen_tries[1]/10 #This is just so that the last result is showing what the acctual power is
# print(gen_tries, abs(start-end))
for generation, i in enumerate(gen_tries):
    i.append(generation)
gen_tries.sort(key=lambda i: i[2], reverse=True) #sorts the list of best solutions for each generation to get which one is the best
for count, i in enumerate(gen_tries):
    gen_tries[count][1] = i[1]/10
for i in gen_tries:
    print(i) #prints the sorted list with the best fitness value in the top, with the size, power, fitness value and generation in it.
print(abs(start-end))

#-----------------------END OF CODE FOR GENETIC ALGORITHM-----------------

#---------------- TRYING TO SOLVE ALL THE CASES---------------------------

#when we have 20 or more different capacites and powers, we have 400 different soltions and

# start_2 = time.time()
# Battery_capacity = list(range(0, 131))  # kWh Trying it out with smaller sample first to look at time
# Battery_power = list(range(0, 131))

# All_solutions = []  
# gen = 0
# for cap in Battery_capacity:
#     for pwr in Battery_power:
#         solution = [cap, pwr, gen]
#         Schedule = ESS_schedule(ESS_capacity_size=cap, ESS_power=pwr,
#                             Energy_hourly_cost=Energy_hourly_cost,
#                             Average_median_cost_day=Average_median_cost_day,
#                             Energy_hourly_use=Energy_hourly_use,
#                             ESS_discharge_eff=ESS_discharge_eff, ESS_charge_eff=ESS_charge_eff)
        
#         solution.append(Fitness_max_saved(Energy_hourly_use=Energy_hourly_use,
#                                           schedule=Schedule,
#                                           Energy_hourly_cost=Energy_hourly_cost,
#                                           ESS_power= pwr,
#                                           ESS_capacity= cap,
#                                           ESS_capacity_cost=ESS_capacity_cost,
#                                           ESS_power_cost=ESS_power_cost,
#                                           ESS_O_and_M_cost=ESS_O_and_M_cost,
#                                           Base_case_cost=Base_case_cost))
#         All_solutions.append(solution)
#         gen += 1
 
# All_solutions.sort(key=lambda i: i[3], reverse=True)        

# end_2 = time.time()
# print(All_solutions, abs(start_2-end_2))


# stuff = fun.Roulette_wheel_selection(np.array(gen_tries))   #to test that the roulette wheel function works (have to be redone.)
# print(stuff)