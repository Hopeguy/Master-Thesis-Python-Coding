
from cProfile import label
from cmath import log
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


Case_2_NPV = pd.read_csv('Results\Divided_cost_results_200_gen\Case_2_NPV_Divided_200_gen.csv')
Case_3_NPV = pd.read_csv('Results\Divided_cost_results_200_gen\Case_3_NPV_Divided_200_gen.csv')

#Sort the dataframe so at index 1 is the solution with the best fitness value (highes NPV)

Case2_NPV_GA = [Case_2_NPV['profit_kWh'][0], -Case_2_NPV['cost_charge'][0], Case_2_NPV['profit_peak_kW'][0], Case_2_NPV['fitness_function'][0], 0]
case3_NPV_GA = [Case_3_NPV['profit_kWh'][0], -Case_3_NPV['cost_charge'][0], Case_3_NPV['profit_peak_kW'][0], Case_3_NPV['fitness_function'][0], Case_3_NPV['profit_saved_gas_heating_total'][0]]

Case2_NPV_FF = [Case_2_NPV['profit_kWh'][1], -Case_2_NPV['cost_charge'][1], Case_2_NPV['profit_peak_kW'][1], Case_2_NPV['fitness_function'][1], 0]
Case3_NPV_FF = [Case_3_NPV['profit_kWh'][1], -Case_3_NPV['cost_charge'][1], Case_3_NPV['profit_peak_kW'][1], Case_3_NPV['fitness_function'][1], Case_3_NPV['profit_saved_gas_heating_total'][1]]

profit_kWh = [Case2_NPV_GA[0], case3_NPV_GA[0], Case2_NPV_FF[0], Case3_NPV_FF[0]]
cost_charge = [Case2_NPV_GA[1], case3_NPV_GA[1], Case2_NPV_FF[1], Case3_NPV_FF[1]]
profit_peak = [Case2_NPV_GA[2], case3_NPV_GA[2], Case2_NPV_FF[2], Case3_NPV_FF[2]]
Fittnes_NPV = [abs(Case2_NPV_GA[3]), case3_NPV_GA[3], abs(Case2_NPV_FF[3]), Case3_NPV_FF[3]]
Saved_cost_of_heating = [Case2_NPV_GA[4], case3_NPV_GA[4], Case2_NPV_FF[4], Case3_NPV_FF[4]]


labels = ['Earnings discharge [Euro]', 'Cost charging [Euro]', 'Earnings peak shaving [Euro]', 'Case 2: -NPV, Case 3: NPV  [Euro]', 'Savings cost of heating [Euro]']
labels_x = ['Case2: GA', 'Case3: GA', 'Case2: FF', 'Case3: FF']
x = np.arange(4)
width = 0.15

fig, ax = plt.subplots()

rect1 = ax.bar(x - width, profit_kWh, width, label = labels[0], log = True)
rect2 = ax.bar(x, cost_charge, width, label = labels[1], log = True)
rect3 = ax.bar(x + width, profit_peak, width, label = labels[2], log = True)
rect4 = ax.bar(x + width + width, Fittnes_NPV, width, label = labels[3])
rect4 = ax.bar(x - width - width, Saved_cost_of_heating, width, label = labels[4], log = True)


ax.set_ylabel('Euro')
ax.set_title('Profits and cost for best solution at 200 iterations')
ax.set_xticks(x)
ax.set_xticklabels(labels_x)
ax.legend(bbox_to_anchor=(1, 1))


plt.savefig('Results\Pictures_etc\cost_divided_NPV_all_cases_new.jpeg',  dpi=300, bbox_inches = "tight")
plt.show