
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

profit_kWh_2 = [Case2_NPV_GA[0], Case2_NPV_FF[0]]
cost_charge_2 = [Case2_NPV_GA[1], Case2_NPV_FF[1]]
profit_peak_2 = [Case2_NPV_GA[2], Case2_NPV_FF[2]]
Fittnes_NPV_2 = [Case2_NPV_GA[3], Case2_NPV_FF[3]]
Saved_cost_of_heating_2 = [Case2_NPV_GA[4], Case2_NPV_FF[4]]
BESS_OPEX_2 = [Case2_NPV_GA[5], Case2_NPV_FF[5]]
ELH_OPEX_2 = [Case2_NPV_GA[6], Case2_NPV_FF[6]]
Cost_investmnet_2 = [Case2_NPV_GA[7], Case2_NPV_FF[7]]

profit_kWh_3 = [case3_NPV_GA[0], Case3_NPV_FF[0]]
cost_charge_3 = [case3_NPV_GA[1], Case3_NPV_FF[1]]
profit_peak_3 = [case3_NPV_GA[2], Case3_NPV_FF[2]]
Fittnes_NPV_3 = [case3_NPV_GA[3], Case3_NPV_FF[3]]
Saved_cost_of_heating_3 = [case3_NPV_GA[4], Case3_NPV_FF[4]]
BESS_OPEX_3 = [case3_NPV_GA[5], Case3_NPV_FF[5]]
ELH_OPEX_3 = [case3_NPV_GA[6], Case3_NPV_FF[6]]
Cost_investmnet_3 = [case3_NPV_GA[7], Case3_NPV_FF[7]]


labels = ['Earnings discharge [Euro]', 'Earnings peak shaving [Euro]', 'Savings cost of heating [Euro]', 'Case 2: -NPV, Case 3: NPV  [Euro]', 'Cost charging [Euro]',   "OPEX BESS [Euro]", "OPEX ELH [Euro]", 'Cost investment [Euro]']
labels_1 = ['Case2: GA', 'Case2: FF']
labels_2 = ['Case3: GA', 'Case3: FF']
x = np.arange(2)
width = 0.3

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10))

rect1 = ax1.bar(x - width, profit_kWh_2, width, label = labels[0])
rect2 = ax1.bar(x - width, profit_peak_2, width, bottom = profit_kWh_2, label = labels[1])
rect3 = ax1.bar(x - width, Saved_cost_of_heating_2, width, bottom = (np.array(profit_kWh_2) + np.array(profit_peak_2)), label = labels[2])
rect4 = ax1.bar(x, Fittnes_NPV_2, width, label = labels[3])
rect5 = ax1.bar(x + width, cost_charge_2, width, label = labels[4])
rect6 = ax1.bar(x + width, BESS_OPEX_2, width, bottom = cost_charge_2,  label = labels[5])
rect7 = ax1.bar(x + width, ELH_OPEX_2, width, bottom = (np.array(cost_charge_2) + np.array(BESS_OPEX_2)), label = labels[6])
rect8 = ax1.bar(x + width, Cost_investmnet_2, width, bottom= (np.array(cost_charge_2) + np.array(BESS_OPEX_2) + np.array(ELH_OPEX_2)), label = labels[7])

rect1 = ax2.bar(x - width, profit_kWh_3, width, label = labels[0])
rect2 = ax2.bar(x - width, profit_peak_3, width, bottom = profit_kWh_3, label = labels[1])
rect3 = ax2.bar(x - width, Saved_cost_of_heating_3, width, bottom = (np.array(profit_kWh_3) + np.array(profit_peak_3)), label = labels[2])
rect4 = ax2.bar(x, Fittnes_NPV_3, width, label = labels[3])
rect5 = ax2.bar(x + width, cost_charge_3, width, label = labels[4])
rect6 = ax2.bar(x + width, BESS_OPEX_3, width, bottom = cost_charge_3,  label = labels[5])
rect7 = ax2.bar(x + width, ELH_OPEX_3, width, bottom = (np.array(cost_charge_3) + np.array(BESS_OPEX_3)), label = labels[6])
rect8 = ax2.bar(x + width, Cost_investmnet_3, width, bottom= (np.array(cost_charge_3) + np.array(BESS_OPEX_3) + np.array(ELH_OPEX_3)), label = labels[7])

ax1.set_ylabel('Euro')
ax2.set_ylabel('Euro - Log scale')
#ax1.set_title('Earnings and cost for best solution at 200 iterations (NPV)', loc='left')
#plt.title('Earnings and cost for best solution at 200 iterations (NPV)', pad= 15)
fig.suptitle('Earnings and cost for best solution at 200 iterations (NPV)', y = 0.93)
ax1.set_xticks(x)
ax2.set_xticks(x)
ax2.set_xticklabels(labels_2)
ax1.set_xticklabels(labels_1)
ax1.legend(loc='upper center', bbox_to_anchor=(1.1, -0.05),
          fancybox=True, shadow=True, ncol=3)


plt.savefig('Results\Pictures_etc\cost_divided_NPV_all_cases_combined_cost_and_earnings.jpeg',  dpi=300, bbox_inches = "tight")
plt.show

