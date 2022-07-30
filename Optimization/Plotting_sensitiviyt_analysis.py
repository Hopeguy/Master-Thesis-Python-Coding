import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



data = pd.read_csv('Results\Sensitivity_analysis_case_2\Sensitivity_results.csv')



X_sensitivity = ['-20%', '-10%', 'Base', '10%', '20%']


Case2_GA_NPV, Case2_GA_LCOS, Case2_FF_NPV, Case2_FF_LCOS = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)
Case2_GA_NPV_STD, Case2_GA_LCOS_STD, Case2_FF_NPV_STD, Case2_FF_LCOS_STD = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)
for count, i in enumerate(X_sensitivity):
    #CASE 2 STD LCOS AND NPV
    Case2_GA_NPV[count] = data.iloc[0][f'{i}']
    Case2_GA_LCOS[count] = data.iloc[1][f'{i}']
    Case2_FF_NPV[count] = data.iloc[2][f'{i}']
    Case2_FF_LCOS[count] = data.iloc[3][f'{i}']

    Case2_GA_NPV_STD[count] = data.iloc[4][f'{i}']
    Case2_GA_LCOS_STD[count] = data.iloc[5][f'{i}']
    Case2_FF_NPV_STD[count] = data.iloc[6][f'{i}']
    Case2_FF_LCOS_STD[count] = data.iloc[7][f'{i}']




fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Change in CAPEX')
ax.set_ylabel('NPV [EURO]')
 

line1, = ax.plot(X_sensitivity, Case2_GA_NPV, label="Case 2 GA NPV", linestyle='dashed', color = "red")
line2, = ax.plot(X_sensitivity, Case2_FF_NPV, label = "Case 2 FF NPV", linewidth = 1, color = 'lawngreen')

ax.legend(loc='right', bbox_to_anchor=(1.5, 1))

ax_2 = ax.twinx()
ax_2.set_ylabel('LCOS [EURO/kWh]')

line3, = ax_2.plot(X_sensitivity, Case2_GA_LCOS, label="Case 2 GA LCOS", linestyle='dotted', color = "cyan")
line4, = ax_2.plot(X_sensitivity, Case2_FF_LCOS, label = "Case 2 FF LCOS", linestyle ='dashdot', color = 'black')

ax_2.legend(loc='right', bbox_to_anchor=(1.5, 0.8))
plt.title('Case 2: NPV and LCOS vs Change in CAPEX')
plt.savefig('Results\Sensitivity_analysis_case_2\Case_2_Sensitivity_NPV_LCOS.jpeg',  dpi=300, bbox_inches = "tight")



