
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

cost_charge_2 = [Case2_LCOS_GA[0], Case2_LCOS_FF[0]]
Fittnes_LCOS_2 = [Case2_LCOS_GA[1], Case2_LCOS_FF[1]]
BESS_OPEX_2 = [Case2_LCOS_GA[2], Case2_LCOS_FF[2]]
Cost_investmnet_2 = [Case2_LCOS_GA[3], Case2_LCOS_FF[3]]
Energy_discharge_2 = [Case2_LCOS_GA[4], Case2_LCOS_FF[4]]

cost_charge_3 = [Case3_LCOS_GA[0], Case3_LCOS_FF[0]]
Fittnes_LCOS_3 = [Case3_LCOS_GA[1], Case3_LCOS_FF[1]]
BESS_OPEX_3 = [Case3_LCOS_GA[2], Case3_LCOS_FF[2]]
Cost_investmnet_3 = [Case3_LCOS_GA[3], Case3_LCOS_FF[3]]
Energy_discharge_3 = [Case3_LCOS_GA[4], Case3_LCOS_FF[4]]

labels = ['Cost charging [Euro]', 'LCOS [Euro/kWh]', "OPEX BESS [Euro]", 'Cost investment [Euro]', 'Discharge electricity [kWh]']
labels_2 = ['Case2: GA', 'Case2: FF']
labels_3 = ['Case3: GA', 'Case3: FF']
x = np.arange(2)
width = 0.3

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,10))

ax3 = ax1.twinx()
ax4 = ax2.twinx()

rect1 = ax1.bar(x - width/2, cost_charge_2, width, label = labels[0])
rect3 = ax1.bar(x - width/2, Cost_investmnet_2, width, bottom= cost_charge_2, label = labels[3])
rect4 = ax1.bar(x - width/2, BESS_OPEX_2, width, bottom= (np.array(cost_charge_2) + np.array(Cost_investmnet_2)), label = labels[2])
#rect2 = ax1.bar(x, Fittnes_LCOS_2, width, label = labels[1])
rect2 = ax3.plot(x, Fittnes_LCOS_2, color= 'black', linewidth=5, label = labels[1])
rect5 = ax1.bar(x + width/2, Energy_discharge_2, width, label = labels[4])

rect1 = ax2.bar(x - width/2, cost_charge_3, width, label = labels[0])
rect3 = ax2.bar(x - width/2, Cost_investmnet_3, width, bottom= cost_charge_3, label = labels[3])
rect4 = ax2.bar(x - width/2, BESS_OPEX_3, width, bottom= (np.array(cost_charge_3) + np.array(Cost_investmnet_3)), label = labels[2])
rect2 = ax4.plot(x, Fittnes_LCOS_3, color= 'black', linewidth=5, label = labels[1])
#rect2 = ax2.bar(x, Fittnes_LCOS_3, width, label = labels[1])
rect5 = ax2.bar(x + width/2, Energy_discharge_3, width, label = labels[4])


ax2.set_yscale('log')
ax1.set_yscale('log')
fig.suptitle('Earnings and cost for best solution at 200 iterations (LCOS)', y = 0.91)
ax1.set_ylabel('Euro or kWh')
ax4.set_ylabel('Euro/kWh')
ax1.set_xticks(x)
ax2.set_xticks(x)
ax1.set_xticklabels(labels_2)
ax2.set_xticklabels(labels_3)
ax1.legend(loc='upper center', bbox_to_anchor=(1.1, -0.05),
          fancybox=True, ncol=3)
plt.legend(loc='upper center', bbox_to_anchor=(-0.13, -0.08), fancybox = False)


plt.savefig('Results\Pictures_etc\cost_divided_LCOS_all_cases_combined_bars.jpeg',  dpi=300, bbox_inches = "tight")
plt.show

