
import numpy as np
import pandas as pd
import numpy_financial as npf
import functions as fun
import time
import matplotlib.pyplot as plt

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.swarm import FA
from opytimizer.spaces import SearchSpace




def fitness_func_NPV(solution):

    """
    Returns the NPV value (used as fitness value in FA)
    """
    

    global cashflow_divided
    global ESS_capacity, ESS_power
    global Schedule_sum
    global Schedule
    global Schedule_capacity
    global ESS_capacity_year
    ESS_capacity, ESS_power, = solution[0]*8000, solution[1]*np.max(Energy_hourly_use)
    
    cashflow_each_year = np.zeros(11)
    cashflow_each_year[0] = -((ESS_capacity_cost*ESS_capacity) + (ESS_power*ESS_power_cost)) #First Year is just capex cost and negative as it is a cost
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
        
        cashflow_each_year[year+1] = Cashflow_yearly[0]
       
    fitness = fun.Fitness_NPV(discount_rate = Discount_rate, cashflows = cashflow_each_year)
    
    return -fitness #negative as the FF algo want to minimize the fitness function


#-------------------------------------------------------end of functions------------------------


# ----------Gets the average cost for each day, and the hourly cost at each hour during the year--------

Electricity_price_read = np.genfromtxt(
    "sto-eur17.csv", delimiter=",")  # Prices in EUR/MWh
El_cost_year = []
El_cost_average_day = []

for i in range(365):
    for k in Electricity_price_read[i][0:24]:
        El_cost_year.append((k/1000)*1.11) #Prices in Euro/kWh, by dividing by 1000, times 1.1 to get 2022 euro values due to inflation
          
    El_cost_average_day.append(((Electricity_price_read[i][24])/1000)*1.1)  #Prices in Euro/kwh that is why we are dividing by 1000, times 1.1 to get 2022 values



# --------------------Read load data for both Electricity and Heating--------
Load_data_read = pd.read_csv("Load_data_electricit_heating_2017.csv", header=0) #Takes values from January, Empty data in 7976 set to 0
Electricity_load_pd = Load_data_read["Electricty [kW]"]

Electricity_load = np.zeros(8760)

for count, i in enumerate(Electricity_load_pd):
    Electricity_load[count] = i


# ---------------Input values (non changable)---------------------------------------'
# ------For Schedule inputs-----------

Energy_hourly_cost = np.array(El_cost_year)     #Prices in Euro/kWh
Average_median_cost_day = np.array(El_cost_average_day) 
Energy_hourly_use = np.array(Electricity_load)
ESS_charge_eff = 0.9
ESS_discharge_eff = 0.9


# Important to note that the maximum SoC for the battery is calculated in the schdule function
# Only import is the Total max size that is also used for calculating the cost


# ------For NPV/LCOE inputs -------------
Lifetime_battery = 10  # in years
ESS_capacity_cost = 427.31   # in Euro(2022) per kWh (CAPEX) all cost included
ESS_power_cost = 1710.2  # in Euro(2022) per kW (all cost included)
Fixed_ESS_O_and_M_cost = 4.19  # in Euro(2022) per kW-year
Variable_ESS_O_and_M_cost = 0.488/1000 # in Euro(2022) per kWh-year 
Discount_rate = 0.08 #8 percent
Peak_cost = 5.92/1.1218 #5.92 dollar (2022) per kW (max per month) change to euro: 1 euro is 1.1218 USD january 1 2022

#------------Setup of parameters for FF algorithm------------

#This is done inside the for loop in order to prevent that values saves between the saves, so it gets set up and restet
#for each run



#-----------set up FF algorithm----------
Result_10_tries = [[],[],[],[],[],[],[],[],[],[],[],[],[]]

for i in range(10):

    start = time.time()
    n = 10 #number of agents (fireflies) Comparable to number of solution 
    lower_bound = [0.1/8000, 0.1/np.max(Energy_hourly_use)]  #first is Capacity, second is for Power they are multiplied with their constant in the fitness function aka 8000, np.max(...), lowest real value is 0.1
    upper_bound = [8000/8000, np.max(Energy_hourly_use)/np.max(Energy_hourly_use)]  #upper bound, first is for Capacity, second is for Power we look here between 1 but in the fitness function these values are times the respective max value
    n_variables = 2 #search space dimension (for this case 2, one for ESS capcity and one for ESS power)
    iteration = 200  #number of iterations the algorithm will run

    
     

    Space = SearchSpace(n, n_variables = n_variables, lower_bound = lower_bound, upper_bound = upper_bound)
    params = {"alpha": 1, "beta": 0.5, "gamma": 1.0}
    optimizer = FA(params = params)

    fitness_function = Function(fitness_func_NPV)     #fitness function to be used

    opt = Opytimizer(space = Space, optimizer = optimizer, function=fitness_function, save_agents=True)
    alh = opt.start(n_iterations = iteration)
    end = time.time()

    
    cost_investment = -(ESS_capacity_cost*opt.space.best_agent.position[0]*8000) - (ESS_power_cost*opt.space.best_agent.position[1]*np.max(Energy_hourly_use))  #Adds the investment cost
    
    Result_10_tries[0].append(opt.space.best_agent.position[0]*8000) #Capacity
    Result_10_tries[1].append(opt.space.best_agent.position[1]*np.max(Energy_hourly_use)) #Power
    Result_10_tries[2].append(-opt.space.best_agent.fit) #fittness function positive when looking for LCOS
    Result_10_tries[3].append(abs(end-start)) #Times
    Result_10_tries[4].append(cashflow_divided[0]) #profit from selling kWh
    Result_10_tries[5].append(cashflow_divided[1])  #profit from Peak_kW
    Result_10_tries[6].append(cashflow_divided[2])  #cost from chargin
    Result_10_tries[7].append(cashflow_divided[3])  #cost OnM fixed
    Result_10_tries[8].append(cashflow_divided[4])  #Cost OnM Variable
    Result_10_tries[9].append(cashflow_divided[5])  #Cashflow_total
    Result_10_tries[10].append(cost_investment) #cost investment
    Result_10_tries[11].append(Schedule_sum[0]) #Charge energy to BESS
    Result_10_tries[12].append(Schedule_sum[1]) #Discharge energy from BESS


Schedule = fun.ESS_schedule(ESS_capacity_size=(opt  .space.best_agent.position[0][0]), ESS_power=(opt.space.best_agent.position[1][0]),
                                        Energy_hourly_cost=Energy_hourly_cost,
                                        Average_median_cost_day=Average_median_cost_day,
                                        Energy_hourly_use=Energy_hourly_use,
                                        ESS_discharge_eff=ESS_discharge_eff, 
                                        ESS_charge_eff=ESS_charge_eff, Year = 10, 
                                        ESS_capacity_prev_year= ESS_capacity_year)

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
                'profit_kWh': Result_10_tries[4],
                'profit_peak_kW': Result_10_tries[5],
                'cost_charge': Result_10_tries[6],
                'cost_O_n_M_fixed': Result_10_tries[7],
                'cost_O_n_m_variable': Result_10_tries[8],
                'Cashflow_total': Result_10_tries[9],
                'Cost_investment': Result_10_tries[10],
                'Summed_charge_kWh': Result_10_tries[11],
                'Summed_Discharge_kWh': Result_10_tries[12]}

df = pd.DataFrame(raw_data, columns = ['ESS_power', 'ESS_capacity', 'fitness_function',
                                           'Time', 'std_ESS_power', 'std_ESS_capacity', 'std_fitness_function',
                                           'profit_kWh', 'profit_peak_kW',
                                           'cost_charge', 'cost_O_n_M_fixed', 'cost_O_n_m_variable', 'Cashflow_total',
                                           'Cost_investment', 'Summed_charge_kWh', 'Summed_Discharge_kWh'])

save_file_name_std = f"Results\Firefly_case_2_ESS_NPV\ESS_power_NPV_etc\Firefly_case_2_ESS_NPV_{iteration}_gen.csv"

save_file_name_schedule = f"Results\Firefly_case_2_ESS_NPV\Charge_discharge_capacity\Firefly_case_2_ESS_NPV_{iteration}_gen_Sch_year_10.csv"

df.to_csv(save_file_name_std, index=False, )
tf.to_csv(save_file_name_schedule, index=False, )


best_agent_pos, best_agent_fit = opt.history.get_convergence("best_agent")

print(
    f"Best agent (position, fit): ({opt.space.best_agent.position}, {-opt.space.best_agent.fit})"
)

#Only plots the last of the ten times it is run

agent_pos_0, agent_fit_0 = opt.history.get_convergence("agents", index =0)
agent_pos_1, agent_fit_1 = opt.history.get_convergence("agents", index =1)
agent_pos_2, agent_fit_2 = opt.history.get_convergence("agents", index =2)
agent_pos_3, agent_fit_3 = opt.history.get_convergence("agents", index =3)
agent_pos_4, agent_fit_4 = opt.history.get_convergence("agents", index =4)
agent_pos_5, agent_fit_5 = opt.history.get_convergence("agents", index =5)
agent_pos_6, agent_fit_6 = opt.history.get_convergence("agents", index =6)
agent_pos_7, agent_fit_7 = opt.history.get_convergence("agents", index =7)
agent_pos_8, agent_fit_8 = opt.history.get_convergence("agents", index =8)
agent_pos_9, agent_fit_9 = opt.history.get_convergence("agents", index =9)


#print("Agent_0", agent_pos_0, agent_fit_0)
#print("Agent_1", agent_pos_1, agent_fit_1)



plt.figure(1)
plt.xlabel("Iteration")
plt.ylabel("Capacity [kWh]")
plt.title("FF Case 2: Capacity when Optimizing for NPV")
plt.plot(range(iteration), agent_pos_0[0]*8000, color = 'green')
plt.plot(range(iteration), agent_pos_1[0]*8000, color = 'red')
plt.plot(range(iteration), agent_pos_2[0]*8000, color = 'blue')
plt.plot(range(iteration), agent_pos_3[0]*8000, color = 'yellow')
plt.plot(range(iteration), agent_pos_4[0]*8000, color = 'black')
plt.plot(range(iteration), agent_pos_5[0]*8000, color = 'magenta')
plt.plot(range(iteration), agent_pos_6[0]*8000, color = 'lightgreen')
plt.plot(range(iteration), agent_pos_7[0]*8000, color = 'olive')
plt.plot(range(iteration), agent_pos_8[0]*8000, color = 'orange')
plt.plot(range(iteration), agent_pos_9[0]*8000, color = 'chocolate')
save_file_name_capacity = f'Results\\Pictures_etc\\Firefly-case-2-NPV-convergence\\Firefly_Case_2_NPV_convergence_capacity_iteration_{iteration}.jpeg'
plt.savefig(save_file_name_capacity, dpi=200, bbox_inches = "tight")
plt.show

plt.figure(2)
plt.xlabel("Iteration")
plt.ylabel("Power [kW]")
plt.title("FF Case 2: Power when Optimizing for NPV")
plt.plot(range(iteration), agent_pos_0[1]*np.max(Energy_hourly_use), color = 'green')
plt.plot(range(iteration), agent_pos_1[1]*np.max(Energy_hourly_use), color = 'red')
plt.plot(range(iteration), agent_pos_2[1]*np.max(Energy_hourly_use), color = 'blue')
plt.plot(range(iteration), agent_pos_3[1]*np.max(Energy_hourly_use), color = 'yellow')
plt.plot(range(iteration), agent_pos_4[1]*np.max(Energy_hourly_use), color = 'black')
plt.plot(range(iteration), agent_pos_5[1]*np.max(Energy_hourly_use), color = 'magenta')
plt.plot(range(iteration), agent_pos_6[1]*np.max(Energy_hourly_use), color = 'lightgreen')
plt.plot(range(iteration), agent_pos_7[1]*np.max(Energy_hourly_use), color = 'olive')
plt.plot(range(iteration), agent_pos_8[1]*np.max(Energy_hourly_use), color = 'orange')
plt.plot(range(iteration), agent_pos_9[1]*np.max(Energy_hourly_use), color = 'chocolate')
save_file_name_power = f'Results\\Pictures_etc\\Firefly-case-2-NPV-convergence\\Firefly_Case_2_NPV_convergence_power_iteration_{iteration}.jpeg'
plt.savefig(save_file_name_power, dpi=200, bbox_inches = "tight")
plt.show

plt.figure(3)
plt.xlabel("Iteration")
plt.ylabel("NPV [Euro]")
plt.title("Firefly Case 2: NPV")
plt.plot(range(iteration), -agent_fit_0, color = 'green')
plt.plot(range(iteration), -agent_fit_1, color = 'red')
plt.plot(range(iteration), -agent_fit_2, color = 'blue')
plt.plot(range(iteration), -agent_fit_3, color = 'yellow')
plt.plot(range(iteration), -agent_fit_4, color = 'black')
plt.plot(range(iteration), -agent_fit_5, color = 'magenta')
plt.plot(range(iteration), -agent_fit_6, color = 'lightgreen')
plt.plot(range(iteration), -agent_fit_7, color = 'olive')
plt.plot(range(iteration), -agent_fit_8, color = 'orange')
plt.plot(range(iteration), -agent_fit_9, color = 'chocolate')

save_file_name_NPV = f"Results\\Pictures_etc\\Firefly-case-2-NPV-convergence\\Firefly_Case_2_NPV_convergence_fitness_iteration_{iteration}.jpeg"
plt.savefig(save_file_name_NPV, dpi=200, bbox_inches = "tight")
plt.show