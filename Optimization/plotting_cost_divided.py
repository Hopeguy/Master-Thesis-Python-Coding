
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




data_NPV = pd.read_csv('Results\Pygad_case_2_ESS_NPV\ESS_power_NPV_etc\Pygad_case_2_NPV_ESS_200_gen.csv')
Case_3_data_NPV = pd.read_csv('Results\Pygad_case_3_ESS_NPV\ESS_power_NPV_etc\Pygad_case_3_ESS_NPV_200_gen.csv')
Case_2_FF_data = pd.read_csv('Results\Firefly_case_2_ESS_NPV\ESS_power_NPV_etc\Firefly_case_2_ESS_NPV_200_gen.csv')
Case_3_FF_data = pd.read_csv('Results\Firefly_case_3_ESS_NPV\ESS_power_NPV_etc\Firefly_case_3_ESS_200_gen.csv')



Case2_NPV = [data_NPV['profit_kWh'].mean(), -data_NPV['cost_charge'].mean(), data_NPV['profit_peak_kW'].mean()]
case3_NPV = [Case_3_data_NPV['profit_kWh'].mean(), -Case_3_data_NPV['cost_charge'].mean(), Case_3_data_NPV['profit_peak_kW'].mean()]

Case2_NPV_FF = [Case_2_FF_data['profit_kWh'].mean(), -Case_2_FF_data['cost_charge'].mean(), Case_2_FF_data['profit_peak_kW'].mean()]
Case3_NPV_FF = [Case_3_FF_data['profit_kWh'].mean(), -Case_3_FF_data['cost_charge'].mean(), Case_3_FF_data['profit_peak_kW'].mean()]

profit_kWh = [Case2_NPV[0], case3_NPV[0], Case2_NPV_FF[0], Case3_NPV_FF[0]]
cost_charge = [Case2_NPV[1], case3_NPV[1], Case2_NPV_FF[1], Case3_NPV_FF[1]]
profit_peak = [Case2_NPV[2], case3_NPV[2], Case2_NPV_FF[1], Case3_NPV_FF[1]]

labels = ['Profit discharge', 'Cost charging', 'Profit peak shaving']
labels_x = ['Case2: GA', 'Case3: GA', 'Case2: FF', 'Case3: FF']
x = np.arange(4)
width = 0.15

fig, ax = plt.subplots()

rect1 = ax.bar(x - width, profit_kWh, width, label = labels[0])
rect2 = ax.bar(x, cost_charge, width, label = labels[1])
rect3 = ax.bar(x + width, profit_peak, width, label = labels[2])

ax.set_ylabel('Euro')
ax.set_title('NPV optimization 200 iterations: Average profits and charging cost')
ax.set_xticks(x)
ax.set_xticklabels(labels_x)
ax.legend(bbox_to_anchor=(1, 1))


plt.savefig('Results\Pictures_etc\Schedule\chargin_cost_profit_peak_discharge_NPV.jpeg',  dpi=300, bbox_inches = "tight")
plt.show
