from ast import Load
from turtle import color
from matplotlib import projections

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('Results/Results_for_printing.csv')
iterations = ['5', '10', '25', '50', '100', '200']

Case2_GA_NPV, Case2_GA_capcity_NPV, Case2_GA_Power_NPV, Case2_GA_time_NPV  = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
Case2_GA_LCOS, Case2_GA_capcity_LCOS, Case2_GA_Power_LCOS, Case2_GA_time_LCOS  = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
Case2_FF_NPV, Case2_FF_capcity_NPV, Case2_FF_Power_NPV, Case2_FF_time_NPV  = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
Case2_FF_LCOS, Case2_FF_capcity_LCOS, Case2_FF_Power_LCOS, Case2_FF_time_LCOS  = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)


Case3_GA_NPV, Case3_GA_capcity_NPV, Case3_GA_Power_NPV, Case3_GA_time_NPV, Case3_GA_ELH_power_NPV  = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
Case3_GA_LCOS, Case3_GA_capcity_LCOS, Case3_GA_Power_LCOS, Case3_GA_time_LCOS, Case3_GA_ELH_power_LCOS  = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
Case3_FF_NPV, Case3_FF_capcity_NPV, Case3_FF_Power_NPV, Case3_FF_time_NPV, Case3_FF_ELH_power_NPV  = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
Case3_FF_LCOS, Case3_FF_capcity_LCOS, Case3_FF_Power_LCOS, Case3_FF_time_LCOS, Case3_FF_ELH_power_LCOS  = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
#------Data for case 2---------
for count, i in enumerate(iterations):
    #For GA Case 2 NPV
    Case2_GA_NPV[count] = data.iloc[0][f'{i}']
    Case2_GA_capcity_NPV[count] = data.iloc[1][f'{i}']
    Case2_GA_Power_NPV[count] = data.iloc[2][f'{i}']
    Case2_GA_time_NPV[count] = data.iloc[3][f'{i}']

    #For GA case 2 LCOS
    Case2_GA_LCOS[count] = data.iloc[4][f'{i}']
    Case2_GA_capcity_LCOS[count] = data.iloc[5][f'{i}']
    Case2_GA_Power_LCOS[count] = data.iloc[6][f'{i}']
    Case2_GA_time_LCOS[count] = data.iloc[7][f'{i}']
    
    #For FF case 2 NPV
    Case2_FF_NPV[count] = data.iloc[8][f'{i}']
    Case2_FF_capcity_NPV[count] = data.iloc[9][f'{i}']
    Case2_FF_Power_NPV[count] = data.iloc[10][f'{i}']
    Case2_FF_time_NPV[count] = data.iloc[11][f'{i}']


    #For FF case 2 LCOS
    Case2_FF_LCOS[count] = data.iloc[12][f'{i}']
    Case2_FF_capcity_LCOS[count] = data.iloc[13][f'{i}']
    Case2_FF_Power_LCOS[count] = data.iloc[14][f'{i}']
    Case2_FF_time_LCOS[count] = data.iloc[15][f'{i}']

for count, i in enumerate(iterations):
    #For GA Case 3 NPV
    Case3_GA_NPV[count] = data.iloc[16][f'{i}']
    Case3_GA_capcity_NPV[count] = data.iloc[17][f'{i}']
    Case3_GA_Power_NPV[count] = data.iloc[18][f'{i}']
    Case3_GA_ELH_power_NPV[count] = data.iloc[19][f'{i}']
    Case3_GA_time_NPV[count] = data.iloc[20][f'{i}']

    #For GA case 3 LCOS
    Case3_GA_LCOS[count] = data.iloc[21][f'{i}']
    Case3_GA_capcity_LCOS[count] = data.iloc[22][f'{i}']
    Case3_GA_Power_LCOS[count] = data.iloc[23][f'{i}']
    Case3_GA_ELH_power_NPV[count] = data.iloc[24][f'{i}']
    Case3_GA_time_LCOS[count] = data.iloc[25][f'{i}']
    
    #For FF case 3 NPV
    Case3_FF_NPV[count] = data.iloc[26][f'{i}']
    Case3_FF_capcity_NPV[count] = data.iloc[27][f'{i}']
    Case3_FF_Power_NPV[count] = data.iloc[28][f'{i}']
    Case3_GA_ELH_power_NPV[count] = data.iloc[29][f'{i}']
    Case3_FF_time_NPV[count] = data.iloc[30][f'{i}']


    #For FF case 3 LCOS
    Case3_FF_LCOS[count] = data.iloc[31][f'{i}']
    Case3_FF_capcity_LCOS[count] = data.iloc[32][f'{i}']
    Case3_FF_Power_LCOS[count] = data.iloc[33][f'{i}']
    Case3_GA_ELH_power_NPV[count] = data.iloc[34][f'{i}']
    Case3_FF_time_LCOS[count] = data.iloc[35][f'{i}']



#------------Combined Case 2 NPV and LCOS for both GA and FF
fig4, ax4 = plt.subplots()

line1, = ax4.plot(iterations, Case2_GA_NPV,  label="GA NPV", linestyle='dashed', color = "red")
line2, = ax4.plot(iterations, Case2_FF_NPV, label="FF NPV", linestyle='dotted', color = "blue")

ax4.set_xlabel('Iterations')
ax4.set_ylabel('NPV [EURO]', color='blue')

ax4_2 = ax4.twinx()
ax4_2.set_ylabel("LCOS [Euro/kWh]", color = 'red')
line3, = ax4_2.plot(iterations, Case2_FF_LCOS, label="FF LCOS", linewidth=1, color='lawngreen')
line4, = ax4_2.plot(iterations, Case2_GA_LCOS, label="GA LCOS", linestyle ='dashdot', color= 'black')

ax4.legend(loc='right', bbox_to_anchor=(1.4, 1))
ax4_2.legend(loc='right', bbox_to_anchor=(1.4, 0.8))
plt.title("Case 2 Average NPV and LCOS vs Iterations")
plt.savefig('Results\Pictures_etc\Compiled Results\Case-2-combined\Case2-NPV-LCOS-Combined-GA-FF.jpeg', dpi=300, bbox_inches = "tight")
plt.show


#---------Combined Case 3 NPV and LCOS for both GA and FF

fig5, ax5 = plt.subplots()

line1, = ax5.plot(iterations, Case3_GA_NPV,  label="GA NPV", linestyle='dashed',  color = "red")
line2, = ax5.plot(iterations, Case3_FF_NPV, label="FF NPV", linestyle='dotted', color = "blue")

ax5.set_xlabel('Iterations')
ax5.set_ylabel('NPV [EURO]', color='blue')

ax5_2 = ax5.twinx()
ax5_2.set_ylabel("LCOS [Euro/kWh]", color = 'red')
line3, = ax5_2.plot(iterations, Case3_FF_LCOS, label="FF LCOS", linewidth=1, color='lawngreen')
line4, = ax5_2.plot(iterations, Case3_GA_LCOS, label="GA LCOS", linestyle ='dashdot',  color= 'black')

ax5.legend(loc='right', bbox_to_anchor=(1.4, 1))
ax5_2.legend(loc='right', bbox_to_anchor=(1.4, 0.8))

plt.title("Case 3 Average NPV and LCOS vs Iterations")
plt.savefig('Results\Pictures_etc\Compiled Results\Case-3-combined\Case3-NPV-LCOS-Combined-GA-FF.jpeg', dpi=300, bbox_inches = "tight")
plt.show




#CASE 3 GA NPV 2D SCATTER PLOT OF CONVERGENCE OF SOLUTIONS

iterations_num = [5, 10, 25, 50, 100, 200]

iter_ALL_combined = []
for count, i in enumerate(iterations):
    Data_GA_case_3_NPV = pd.read_csv(f'Results\Pygad_case_3_ESS_NPV\ESS_power_NPV_etc\Pygad_case_3_ESS_NPV_{i}_gen.csv')
    Power = Data_GA_case_3_NPV["ESS_power"].values
    Capacity = Data_GA_case_3_NPV["ESS_capacity"].values
    Power_ELH = Data_GA_case_3_NPV["ELH_power"].values
    NPV = Data_GA_case_3_NPV["fitness_function"].values
    combined = [NPV, Capacity, Power, Power_ELH]
    iter_ALL_combined.append(combined)


fig6, ax6 = plt.subplots()

Scatter_X_all = []
Scatter_NPV = []
Scatter_Capacity = []
Scatter_Power = []
Scatter_ELH_Power = []
for count, i in enumerate(iterations_num):
    for num in range(10):
        Scatter_X_all.append(i)
        Scatter_NPV.append(iter_ALL_combined[count][0][num]) #Takes all the NPV values
        Scatter_Capacity.append(iter_ALL_combined[count][1][num])
        Scatter_Power.append(iter_ALL_combined[count][2][num])
        Scatter_ELH_Power.append(iter_ALL_combined[count][3][num])


ax6.scatter(Scatter_X_all, Scatter_NPV, label = "NPV [EURO]", color = "blue") #Takes out the NPV
    

ax6.set_ylabel("NPV [EURO]")
ax6.set_xlabel('Iterations')
ax6_2 = ax6.twinx()
ax6_2.set_ylabel("kWh or kW")
ax6.set_xticks(iterations_num)

ax6_2.scatter(Scatter_X_all, Scatter_Capacity, label = "BESS Capacity [kWh]", color = "red") #plots the Capacity scatter
ax6_2.scatter(Scatter_X_all, Scatter_Power, label = "BESS Power [kW]", color = "magenta")     #Plots the BESS power scatter
ax6_2.scatter(Scatter_X_all, Scatter_ELH_Power, label = "ELH Power [kW]", color = "lawngreen") #plots the ELH power
ax6.legend(loc='right', bbox_to_anchor=(1.4, 1))
ax6_2.legend(loc='right', bbox_to_anchor=(1.54, 0.8))

plt.title("Case 3 NPV GA: Capacity, Power and ELH Power vs Iterations")
plt.savefig('Results\Pictures_etc\Compiled Results\Case-3-GA\Case3-NPV-GA-Scatter-Convergence.jpeg', dpi=300, bbox_inches = "tight")
plt.show




#CASE 3 FF NPV 2D SCATTER PLOT OF CONVERGENCE OF SOLUTIONS

iterations_num = [5, 10, 25, 50, 100, 200]

iter_ALL_combined = []
for count, i in enumerate(iterations):
    Data_FF_case_3_NPV = pd.read_csv(f'Results\Firefly_case_3_ESS_NPV\ESS_power_NPV_etc\Firefly_case_3_ESS_{i}_gen.csv')
    Power = Data_FF_case_3_NPV["ESS_power"].values
    Capacity = Data_FF_case_3_NPV["ESS_capacity"].values
    Power_ELH = Data_FF_case_3_NPV["ELH_power"].values
    NPV = Data_FF_case_3_NPV["fitness_function"].values
    combined = [NPV, Capacity, Power, Power_ELH]
    iter_ALL_combined.append(combined)


fig7, ax7 = plt.subplots()

Scatter_X_all = []
Scatter_NPV = []
Scatter_Capacity = []
Scatter_Power = []
Scatter_ELH_Power = []
for count, i in enumerate(iterations_num):
    for num in range(10):
        Scatter_X_all.append(i)
        Scatter_NPV.append(iter_ALL_combined[count][0][num]) #Takes all the NPV values
        Scatter_Capacity.append(iter_ALL_combined[count][1][num])
        Scatter_Power.append(iter_ALL_combined[count][2][num])
        Scatter_ELH_Power.append(iter_ALL_combined[count][3][num])


ax7.scatter(Scatter_X_all, Scatter_NPV, label = "NPV [EURO]", color = "blue") #Takes out the NPV
    

ax7.set_ylabel("NPV [EURO]")
ax7.set_xlabel('Iterations')
ax7_2 = ax7.twinx()
ax7_2.set_ylabel("kWh or kW")
ax7.set_xticks(iterations_num)


ax7_2.scatter(Scatter_X_all, Scatter_Capacity, label = "BESS Capacity [kWh]", color = "red") #plots the Capacity scatter
ax7_2.scatter(Scatter_X_all, Scatter_Power, label = "BESS Power [kW]", color = "magenta")     #Plots the BESS power scatter
ax7_2.scatter(Scatter_X_all, Scatter_ELH_Power, label = "ELH Power [kW]", color = "lawngreen") #plots the ELH power
ax7.legend(loc='right', bbox_to_anchor=(1.4, 1))
ax7_2.legend(loc='right', bbox_to_anchor=(1.54, 0.8))

plt.title("Case 3 NPV FF: Capacity, Power and ELH Power vs Iterations")
plt.savefig('Results\Pictures_etc\Compiled Results\Case-3-FF\Case3-NPV-FF-Scatter-Convergence.jpeg', dpi=300, bbox_inches = "tight")
plt.show


#CASE 3 GA LCOS 2D SCATTER PLOT OF CONVERGENCE OF SOLUTIONS

iterations_num = [5, 10, 25, 50, 100, 200]

iter_ALL_combined = []
for count, i in enumerate(iterations):
    Data_GA_case_3_LCOS = pd.read_csv(f'Results\Pygad_case_3_ESS_LCOS\ESS_power_LCOS_etc\Pygad_case_3_ESS_LCOS_{i}_gen.csv')
    Power = Data_GA_case_3_LCOS["ESS_power"].values
    Capacity = Data_GA_case_3_LCOS["ESS_capacity"].values
    Power_ELH = Data_GA_case_3_LCOS["ELH_power"].values
    LCOS = Data_GA_case_3_LCOS["fitness_function"].values
    combined = [LCOS, Capacity, Power, Power_ELH]
    iter_ALL_combined.append(combined)


fig8, ax8 = plt.subplots()

Scatter_X_all = []
Scatter_LCOS = []
Scatter_Capacity = []
Scatter_Power = []
Scatter_ELH_Power = []
for count, i in enumerate(iterations_num):
    for num in range(10):
        Scatter_X_all.append(i)
        Scatter_LCOS.append(iter_ALL_combined[count][0][num]) #Takes all the NPV values
        Scatter_Capacity.append(iter_ALL_combined[count][1][num])
        Scatter_Power.append(iter_ALL_combined[count][2][num])
        Scatter_ELH_Power.append(iter_ALL_combined[count][3][num])


ax8.scatter(Scatter_X_all, Scatter_LCOS, label = "LCOS [EURO/kWh]", color = "blue") #Takes out the NPV
    

ax8.set_ylabel("LCOS [EURO/kWh]")
ax8.set_xlabel('Iterations')
ax8_2 = ax8.twinx()
ax8_2.set_ylabel("kWh or kW")
ax8.set_xticks(iterations_num)


ax8_2.scatter(Scatter_X_all, Scatter_Capacity, label = "BESS Capacity [kWh]", color = "red") #plots the Capacity scatter
ax8_2.scatter(Scatter_X_all, Scatter_Power, label = "BESS Power [kW]", color = "magenta")     #Plots the BESS power scatter
ax8_2.scatter(Scatter_X_all, Scatter_ELH_Power, label = "ELH Power [kW]", color = "lawngreen") #plots the ELH power
ax8.legend(loc='right', bbox_to_anchor=(1.505, 1))
ax8_2.legend(loc='right', bbox_to_anchor=(1.54, 0.8))

plt.title("Case 3 LCOS GA: Capacity, Power and ELH Power vs Iterations")
plt.savefig('Results\Pictures_etc\Compiled Results\Case-3-GA\Case3-LCOS-GA-Scatter-Convergence.jpeg', dpi=300, bbox_inches = "tight")
plt.show


#CASE 3 FF LCOS 2D SCATTER PLOT OF CONVERGENCE OF SOLUTIONS

iterations_num = [5, 10, 25, 50, 100, 200]

iter_ALL_combined = []
for count, i in enumerate(iterations):
    Data_FF_case_3_LCOS = pd.read_csv(f'Results\Firefly_case_3_ESS_LCOS\ESS_power_LCOS_etc\Firefly_case_3_ESS_LCOS_{i}_gen.csv')
    Power = Data_FF_case_3_LCOS["ESS_power"].values
    Capacity = Data_FF_case_3_LCOS["ESS_capacity"].values
    Power_ELH = Data_FF_case_3_LCOS["ELH_power"].values
    LCOS = Data_FF_case_3_LCOS["fitness_function"].values
    combined = [LCOS, Capacity, Power, Power_ELH]
    iter_ALL_combined.append(combined)


fig9, ax9 = plt.subplots()

Scatter_X_all = []
Scatter_LCOS = []
Scatter_Capacity = []
Scatter_Power = []
Scatter_ELH_Power = []
for count, i in enumerate(iterations_num):
    for num in range(10):
        Scatter_X_all.append(i)
        Scatter_LCOS.append(iter_ALL_combined[count][0][num]) #Takes all the NPV values
        Scatter_Capacity.append(iter_ALL_combined[count][1][num])
        Scatter_Power.append(iter_ALL_combined[count][2][num])
        Scatter_ELH_Power.append(iter_ALL_combined[count][3][num])


ax9.scatter(Scatter_X_all, Scatter_LCOS, label = "LCOS [EURO/kWh]", color = "blue") #Takes out the NPV
    

ax9.set_ylabel("LCOS [EURO/kWh]")
ax9.set_xlabel('Iterations')
ax9_2 = ax9.twinx()
ax9_2.set_ylabel("kWh or kW")
ax9.set_xticks(iterations_num)


ax9_2.scatter(Scatter_X_all, Scatter_Capacity, label = "BESS Capacity [kWh]", color = "red") #plots the Capacity scatter
ax9_2.scatter(Scatter_X_all, Scatter_Power, label = "BESS Power [kW]", color = "magenta")     #Plots the BESS power scatter
ax9_2.scatter(Scatter_X_all, Scatter_ELH_Power, label = "ELH Power [kW]", color = "lawngreen") #plots the ELH power
ax9.legend(loc='right', bbox_to_anchor=(1.505, 1))
ax9_2.legend(loc='right', bbox_to_anchor=(1.54, 0.8))

plt.title("Case 3 LCOS FF: Capacity, Power and ELH Power vs Iterations")
plt.savefig('Results\Pictures_etc\Compiled Results\Case-3-FF\Case3-LCOS-FF-Scatter-Convergence.jpeg', dpi=300, bbox_inches = "tight")
plt.show