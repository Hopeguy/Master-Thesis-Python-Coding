
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


Case_2_NPV = pd.read_csv('Results\Divided_cost_results_200_gen\Case_2_NPV_Divided_200_gen.csv')
Case_3_NPV = pd.read_csv('Results\Divided_cost_results_200_gen\Case_3_NPV_Divided_200_gen.csv')

#Sort the dataframe so at index 1 is the solution with the best fitness value (highes NPV)

Case2_NPV_GA = [Case_2_NPV['profit_kWh'][0], -Case_2_NPV['cost_charge'][0], Case_2_NPV['profit_peak_kW'][0], Case_2_NPV['fitness_function'][0], 0, -(Case_2_NPV['cost_O_n_M_fixed'][0] + Case_2_NPV['cost_O_n_m_variable'][0]), 0, -Case_2_NPV['Cost_investment'][0]]
case3_NPV_GA = [Case_3_NPV['profit_kWh'][0], -Case_3_NPV['cost_charge'][0], Case_3_NPV['profit_peak_kW'][0], Case_3_NPV['fitness_function'][0], Case_3_NPV['profit_saved_gas_heating_total'][0], -(Case_3_NPV['cost_O_n_M_fixed'][0] + Case_3_NPV['cost_O_n_m_variable'][0]), -Case_3_NPV['OPEX_ELH'][0], Case_3_NPV['Cost_investment'][0]]


Case2_NPV_FF = [Case_2_NPV['profit_kWh'][1], -Case_2_NPV['cost_charge'][1], Case_2_NPV['profit_peak_kW'][1], Case_2_NPV['fitness_function'][1], 0,  -(Case_2_NPV['cost_O_n_M_fixed'][1] + Case_2_NPV['cost_O_n_m_variable'][1]), 0, -Case_2_NPV['Cost_investment'][1]]
Case3_NPV_FF = [Case_3_NPV['profit_kWh'][1], -Case_3_NPV['cost_charge'][1], Case_3_NPV['profit_peak_kW'][1], Case_3_NPV['fitness_function'][1], Case_3_NPV['profit_saved_gas_heating_total'][1], -(Case_3_NPV['cost_O_n_M_fixed'][1] + Case_3_NPV['cost_O_n_m_variable'][1]), -Case_3_NPV['OPEX_ELH'][1], Case_3_NPV['Cost_investment'][1]]

profit_kWh = [Case2_NPV_GA[0], case3_NPV_GA[0], Case2_NPV_FF[0], Case3_NPV_FF[0]]
cost_charge = [Case2_NPV_GA[1], case3_NPV_GA[1], Case2_NPV_FF[1], Case3_NPV_FF[1]]
profit_peak = [Case2_NPV_GA[2], case3_NPV_GA[2], Case2_NPV_FF[2], Case3_NPV_FF[2]]
Fittnes_NPV = [abs(Case2_NPV_GA[3]), case3_NPV_GA[3], abs(Case2_NPV_FF[3]), Case3_NPV_FF[3]]
Saved_cost_of_heating = [Case2_NPV_GA[4], case3_NPV_GA[4], Case2_NPV_FF[4], Case3_NPV_FF[4]]
BESS_OPEX = [Case2_NPV_GA[5], case3_NPV_GA[5], Case2_NPV_FF[5], Case3_NPV_FF[5]]
ELH_OPEX = [Case2_NPV_GA[6], case3_NPV_GA[6], Case2_NPV_FF[6], Case3_NPV_FF[6]]
Cost_investmnet = [Case2_NPV_GA[7], case3_NPV_GA[7], Case2_NPV_FF[7], Case3_NPV_FF[7]]


labels = ['Earnings discharge [Euro]', 'Earnings peak shaving [Euro]', 'Savings cost of heating [Euro]', 'Case 2: -NPV, Case 3: NPV  [Euro]', 'Cost charging [Euro]',   "OPEX BESS [Euro]", "OPEX ELH [Euro]", 'Cost investment [Euro]']
labels_x = ['Case2: GA', 'Case3: GA', 'Case2: FF', 'Case3: FF']
x = np.arange(4)
width = 0.110

fig, ax = plt.subplots()

rect1 = ax.bar(x - width - width - width, profit_kWh, width, label = labels[0], log = True)
rect2 = ax.bar(x - width - width, profit_peak, width, label = labels[1], log = True)
rect3 = ax.bar(x - width, Saved_cost_of_heating, width, label = labels[2], log = True)
rect4 = ax.bar(x, Fittnes_NPV, width, label = labels[3], log = True)
rect5 = ax.bar(x + width, cost_charge, width, label = labels[4], log = True)
rect6 = ax.bar(x + width + width, BESS_OPEX, width, label = labels[5], log = True)
rect7 = ax.bar(x + width + width + width, ELH_OPEX, width, label = labels[6], log = True)
rect8 = ax.bar(x + width + width + width + width, Cost_investmnet, width, label = labels[7], log = True)


ax.set_ylabel('Euro')
ax.set_title('Earnings and cost for best solution at 200 iterations (NPV)')
ax.set_xticks(x)
ax.set_xticklabels(labels_x)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=3)


plt.savefig('Results\Pictures_etc\cost_divided_NPV_all_cases_new.jpeg',  dpi=300, bbox_inches = "tight")
plt.show


#-----Calculating percentage of each cost/earnings----------¨

Case = ['Case2: GA', 'Case3: GA', 'Case2: FF', 'Case3: FF']

Total_profits = np.zeros(4)
Saved_heating_percentage = []

for count, i in enumerate(profit_kWh):
    Total_profits[count] += i + profit_peak[count] + Saved_cost_of_heating[count]
    Saved_heating_percentage.append(Saved_cost_of_heating[count]/Total_profits[count])

for percentage in Saved_heating_percentage:
    print('Percentage profit saved heating: ', percentage*100, '%')
