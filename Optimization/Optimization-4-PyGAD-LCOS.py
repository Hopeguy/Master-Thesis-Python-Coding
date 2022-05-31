
import pandas as pd
import numpy as np
import numpy_financial as npf
import functions as fun
import time
import pygad
import matplotlib.pyplot as plt

def fitness_func_LCOS(solution, solution_idx):
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


def callback_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])
    #print("population: ", ga_instance.population)


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
ESS_capacity_cost = 389.2   # in Euro(2022) per kWh (CAPEX) all cost included
ESS_power_cost = 148.8  # in Euro(2022) per kW (all cost included)
Fixed_ESS_O_and_M_cost = 4.19  # in Euro(2022) per kW-year
Variable_ESS_O_and_M_cost = 0.488/1000 # in Euro(2022) per kWh-year 
Discount_rate = 0.08 #8 percent   
Peak_cost = 5.92/1.1218 #5.92 dollar (2022) per kW (max per month) change to euro: 1 euro is 1.1218 USD january 1 2022


### --------Preparing other varuables---------

fitness_function = fitness_func_LCOS  #CHANGE BETWEEN LCOS OR NPV AS FITNESS FUNCTION

sol_per_pop = 50          #Number of solutions per population, Comparable to "agents in FF" "Good results at 50 // 200 generation"
num_generations = 200     #number of generation to run the algorithm
num_parents_mating = int(sol_per_pop/2)     #number of solutions that will be mating (50% of total solutions used each generation)
init_range_low = 0.1          #lowest value starting solutions can take #not used
init_range_high = 2000     #highest value starting solutions can take #not used

parent_selection_type = "rank"      #Method choice for how to pick parent, can be: [sss, rws, sus, rank, random, tournament]
keep_parents = -1       #Keeps all parents into the next generation (this is in order to not forget good solutions)

crossover_type = "uniform"      #method to crossover the genetics between the two parents, can be [singe_point, two_points, uniform, scattered, ]
crossover_probability = 0.8     #How likely it is for a parent to do a crossover, 0.8 is equal to 80%

mutation_type = "random"        #what operation the mutation will take, can be [random, swap, adaptive]
mutation_probability = 0.1       # 10 percent chance of mutation operation to happen for a solution
gene_space = [{'low': 0.1, 'high': 8000}, {'low': 0.1, 'high': np.max(Electricity_load)}] #8000 kWh as max as it is aournd 4 hours system


#---------------------------------------------


Result_10_tries = [[],[],[],[],[],[],[],[],[],[],[]]

for i in range(10):
    #-----------Set up ga-------------
    ga_instance = pygad.GA(num_generations=num_generations,
                       allow_duplicate_genes= False,
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
                       save_solutions=True,
                       save_best_solutions=True,
                       suppress_warnings=True) 
    start = time.time() 
    ga_instance.run()
    end = time.time()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    cost_investment = -((ESS_capacity_cost*solution[0]) + (solution[1]*ESS_power_cost))
    Result_10_tries[0].append(solution[0]) #Capacity
    Result_10_tries[1].append(solution[1]) #Power
    Result_10_tries[2].append(-solution_fitness) #fittness function Negative when looking at LCOS
    Result_10_tries[3].append(abs(end-start)) #Times
    Result_10_tries[4].append(cashflow_divided[0]) #charge cost
    Result_10_tries[5].append(cashflow_divided[1])  #FIxed OnM cost
    Result_10_tries[6].append(cashflow_divided[2])  #Variable OnM cost
    Result_10_tries[7].append(cashflow_divided[3])  #total cost combined
    Result_10_tries[8].append(cost_investment) #cost investment
    Result_10_tries[9].append(Schedule_sum[0]) #Charge energy to BESS
    Result_10_tries[10].append(Schedule_sum[1]) #Discharge energy from BESS

    print(Result_10_tries[2])


Schedule = fun.ESS_schedule(ESS_capacity_size=solution[0], ESS_power=solution[1],  #Schedule function see function for more details
                                        Energy_hourly_cost=Energy_hourly_cost,
                                        Average_median_cost_day=Average_median_cost_day,
                                        Energy_hourly_use=Energy_hourly_use,
                                        ESS_discharge_eff=ESS_discharge_eff,
                                        ESS_charge_eff=ESS_charge_eff,
                                        Year = 10,
                                        ESS_capacity_prev_year = ESS_capacity_year)


Average_hourly_cost = []
for count,average in enumerate(Average_median_cost_day):
    for i in range(24):
        Average_hourly_cost.append(average)
        

Charge_discharge = []

for count, hour_cost in enumerate(Energy_hourly_cost):
    if hour_cost < Average_hourly_cost[count]:
        Charge_discharge.append('Charge+++++')
    else:
        Charge_discharge.append('Discharge-----')

Schedule_last_year = [[],[],[],[],[],[],[]]
Schedule_last_year[0] = Schedule[0][:,0]
Schedule_last_year[1] = Schedule[0][:,1]
Schedule_last_year[2] = Schedule[2]
Schedule_last_year[3] = Energy_hourly_use
Schedule_last_year[4] = np.array(Average_hourly_cost)
Schedule_last_year[5] = np.array(El_cost_year)
Schedule_last_year[6] = Charge_discharge

data_schedule = {'charge': Schedule_last_year[0], 'discharge': Schedule_last_year[1],
             'capacity': Schedule_last_year[2], 'Energy_use': Schedule_last_year[3], 
             'Average_daily_cost': Schedule_last_year[4], 'Energy_hourly_cost':Schedule_last_year[5],
             'Charge_or_discharge': Schedule_last_year[6]}

tf = pd.DataFrame(data_schedule, columns=['charge', 'discharge', 'capacity',
                             'Energy_use', 'Average_daily_cost', 'Energy_hourly_cost', 'Charge_or_discharge'])



std_ESS_power = np.std(Result_10_tries[1]) 
std_ESS_capacity = np.std(Result_10_tries[0])
std_fitness_function = np.std(Result_10_tries[2])


raw_data = {'ESS_power': Result_10_tries[1],
                'ESS_capacity': Result_10_tries[0],
                'fitness_function': Result_10_tries[2],
                'Time': Result_10_tries[3],
                'std_ESS_power': std_ESS_power,
                'std_ESS_capacity': std_ESS_capacity,
                'std_fitness_function': std_fitness_function,
                'Charge_cost': Result_10_tries[4],
                'Fixed_OnM': Result_10_tries[5],
                'Variable_OnM': Result_10_tries[6],
                'Combined_cost_charge': Result_10_tries[7],
                'Cost_investment': Result_10_tries[8],
                'Summed_charge_kWh': Result_10_tries[9],
                'Summed_Discharge_kWh': Result_10_tries[10]}

df = pd.DataFrame(raw_data, columns = ['ESS_power', 'ESS_capacity', 'fitness_function',
                                           'Time', 'std_ESS_power', 'std_ESS_capacity', 'std_fitness_function',
                                           'Charge_cost', 'Fixed_OnM',
                                           'Variable_OnM', 'Combined_cost_charge', 'Cost_investment', 'Summed_charge_kWh',
                                           'Summed_Discharge_kWh'])


save_file_name_std = f"Results\Pygad_case_2_ESS_LCOS\ESS_power_LCOS_etc\Pygad_case_2_LCOS_ESS_{num_generations}_gen.csv"
save_file_name_schedule = f"Results\Pygad_case_2_ESS_LCOS\Charge_discharge_capacity\Pygad_case_2_LCOS_ESS_{num_generations}_gen_Sch_year_10.csv"


df.to_csv(save_file_name_std, index=False, )
tf.to_csv(save_file_name_schedule, index=False)


save_file_name_fittnes_plot = f"Results\\Pictures_etc\\Pygad-case-2-LCOS-convergence\\fitness_LCOS_over_generation_{num_generations}_gen.jpeg"
save_file_name_genes_plot = f"Results\\Pictures_etc\\Pygad-case-2-LCOS-convergence\\Best_genes_LCOS_{num_generations}_gen.jpeg"
save_file_name_solution_rate_plot = f"Results\\Pictures_etc\\Pygad-case-2-LCOS-convergence\\New_Solution_rate_LCOS_{num_generations}_gen.jpeg"

#The plot for the fitness value is reveresed as the LCOS values are supposed to be positive but as the GA
#wants to maximize the fittness function the negative value is given in the fitness function
#therefore the plot results in reverse and the values are supposed to be positive and the line inversed.

ga_instance.plot_fitness(title= "GA Case 2: LCOS", xlabel= "Generation",
                ylabel="LCOS [Euro/kWh]", plot_type="plot", save_dir=save_file_name_fittnes_plot)
ga_instance.plot_genes(graph_type='plot', title= "GA case 2: Best Genes LCOS", solutions="best", save_dir=save_file_name_genes_plot)
ga_instance.plot_new_solution_rate(title = "GA Case 2: New solution rate vs Generation LCOS", ylabel= "New solutions explored", save_dir = save_file_name_solution_rate_plot)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(solution, -solution_fitness)