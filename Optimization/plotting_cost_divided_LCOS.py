
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


Case_2_LCOS = pd.read_csv('Results\Divided_cost_results_200_gen\Case_2_LCOS_Divided_200_gen.csv')
Case_3_LCOS = pd.read_csv('Results\Divided_cost_results_200_gen\Case_3_LCOS_Divided_200_gen.csv')

#Sort the dataframe so at index 1 is the solution with the best fitness value (highes NPV)

Case2_LCOS_GA = [Case_2_LCOS['cost_charge'][0], Case_2_LCOS['fitness_function'][0], (Case_2_LCOS['cost_O_n_M_fixed'][0] + Case_2_LCOS['cost_O_n_m_variable'][0]), -Case_2_LCOS['Cost_investment'][0], Case_2_LCOS["Summed_Discharge_kWh"][0]]
Case3_LCOS_GA = [Case_3_LCOS['cost_charge'][0], Case_3_LCOS['fitness_function'][0], (Case_3_LCOS['cost_O_n_M_fixed'][0] + Case_3_LCOS['cost_O_n_m_variable'][0]), -Case_3_LCOS['Cost_investment'][0], Case_3_LCOS["Summed_Discharge_kWh"][0]]

Case2_LCOS_FF = [Case_2_LCOS['cost_charge'][1], Case_2_LCOS['fitness_function'][1], (Case_2_LCOS['cost_O_n_M_fixed'][1] + Case_2_LCOS['cost_O_n_m_variable'][1]), -Case_2_LCOS['Cost_investment'][1], Case_2_LCOS["Summed_Discharge_kWh"][1]]
Case3_LCOS_FF = [Case_3_LCOS['cost_charge'][1], Case_3_LCOS['fitness_function'][1], (Case_3_LCOS['cost_O_n_M_fixed'][1] + Case_3_LCOS['cost_O_n_m_variable'][1]), -Case_3_LCOS['Cost_investment'][1], Case_3_LCOS["Summed_Discharge_kWh"][1]]

cost_charge = [Case2_LCOS_GA[0], Case3_LCOS_GA[0], Case2_LCOS_FF[0], Case3_LCOS_FF[0]]
Fittnes_LCOS = [Case2_LCOS_GA[1], Case3_LCOS_GA[1], Case2_LCOS_FF[1], Case3_LCOS_FF[1]]
BESS_OPEX = [Case2_LCOS_GA[2], Case3_LCOS_GA[2], Case2_LCOS_FF[2], Case3_LCOS_FF[2]]
Cost_investmnet = [Case2_LCOS_GA[3], Case3_LCOS_GA[3], Case2_LCOS_FF[3], Case3_LCOS_FF[3]]
Energy_discharge = [Case2_LCOS_GA[4], Case3_LCOS_GA[4], Case2_LCOS_FF[4], Case3_LCOS_FF[4]]


labels = ['Cost charging [Euro]', 'LCOS [Euro/kWh]', "OPEX BESS [Euro]", 'Cost investment [Euro]', 'Discharge electricity [kWh]']
labels_x = ['Case2: GA', 'Case3: GA', 'Case2: FF', 'Case3: FF']
x = np.arange(4)
width = 0.125

fig, ax = plt.subplots()

rect1 = ax.bar(x - width - width, cost_charge, width, label = labels[0])
rect2 = ax.bar(x, Fittnes_LCOS, width, label = labels[1])
rect3 = ax.bar(x - width, Cost_investmnet, width, label = labels[3])
rect4 = ax.bar(x + width, BESS_OPEX, width, label = labels[2])
rect5 = ax.bar(x + width + width, Energy_discharge, width, label = labels[4])


ax.set_yscale('log')
ax.set_ylabel('Euro or kWh')
ax.set_title('Earnings and cost for best solution at 200 iterations (LCOS)')
ax.set_xticks(x)
ax.set_xticklabels(labels_x)
ax.legend(bbox_to_anchor=(1, 1))


plt.savefig('Results\Pictures_etc\cost_divided_LCOS_all_cases_new.jpeg',  dpi=300, bbox_inches = "tight")
plt.show


#-----Calculating percentage of each cost/earnings----------¨

Case = ['Case2: GA', 'Case3: GA', 'Case2: FF', 'Case3: FF']

Total_profits = np.zeros(4)
Cost_investment_percentage = []

for count, i in enumerate(cost_charge):
    Total_profits[count] += i + BESS_OPEX[count] + Cost_investmnet[count]
    Cost_investment_percentage.append(Cost_investmnet[count]/Total_profits[count])

for percentage in Cost_investment_percentage:
    print('cost investment: ', percentage*100, '%')
