
from ast import Load
from turtle import color
from matplotlib import projections

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#-------CASE 2 GA--------------

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


#CASE 2 FF NPV 3D SCATTER PLOT OF CONVERGENCE OF SOLUTIONS

iter_ALL_combined = []
for count, i in enumerate(iterations):
    Data_FF_case_2_NPV = pd.read_csv(f'Results\Firefly_case_2_ESS_NPV\ESS_power_NPV_etc\Firefly_case_2_ESS_NPV_{i}_gen.csv')
    Power = Data_FF_case_2_NPV["ESS_power"].values
    Capacity = Data_FF_case_2_NPV["ESS_capacity"].values
    NPV = (Data_FF_case_2_NPV["fitness_function"].values)*1000
    combined = [NPV, Capacity, Power]
    iter_ALL_combined.append(combined)


All = [(iter_ALL_combined[0],'^', 'r', '5 gen'),(iter_ALL_combined[1],'p', 'lawngreen', '10 gen'),
(iter_ALL_combined[2], 'X', 'g', '25 gen'), (iter_ALL_combined[3], 'D', 'b', '50 gen'),
(iter_ALL_combined[4], '1', 'coral', '100 gen'), (iter_ALL_combined[5], '+', 'olive', '200 gen')]


fig1 = plt.figure(3)
ax1 = fig1.add_subplot(projection='3d')
for t, m, c, label in All:
    xs = t[2]
    ys = t[1]
    zs = t[0]
    ax1.scatter(xs, ys, zs, marker = m, c = c, label = label)
fig1.tight_layout()
fig1.subplots_adjust(right=0.9)
ax1.legend(loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=12)
ax1.view_init(10, 45)
ax1.set_xlabel('Power [kW]')
ax1.set_ylabel('Capacity [kWh]')
ax1.set_zlabel('NPV [EURO]', labelpad=15)
ax1.set_title("Case 2 FF: Optimal NPV convergence vs Iterations")



plt.savefig('Results\Pictures_etc\Compiled Results\Case-2-FF\Case2-FF-Optimal-NPV-vs-iterations.jpeg',  dpi=300, bbox_inches = "tight")
plt.show()




#--------CASE 2 FF LCOS 3D SCATTER PLOT of Convergence of solutions

iter_ALL_combined = []
for count, i in enumerate(iterations):
    Data_FF_case_2_LCOS = pd.read_csv(f'Results\Firefly_case_2_ESS_LCOS\ESS_power_LCOS_etc\Firefly_case_2_ESS_LCOS_{i}_gen.csv')
    Power = Data_FF_case_2_LCOS["ESS_power"].values
    Capacity = Data_FF_case_2_LCOS["ESS_capacity"].values
    LCOS = (Data_FF_case_2_LCOS["fitness_function"].values)*1000
    combined = [LCOS, Capacity, Power]
    iter_ALL_combined.append(combined)


All = [(iter_ALL_combined[0],'^', 'r', '5 gen'),(iter_ALL_combined[1],'p', 'lawngreen', '10 gen'),
(iter_ALL_combined[2], 'X', 'g', '25 gen'), (iter_ALL_combined[3], 'D', 'b', '50 gen'),
(iter_ALL_combined[4], '1', 'coral', '100 gen'), (iter_ALL_combined[5], '+', 'olive', '200 gen')]


fig1 = plt.figure(3)
ax1 = fig1.add_subplot(projection='3d')
for t, m, c, label in All:
    xs = t[2]
    ys = t[1]
    zs = t[0]
    ax1.scatter(xs, ys, zs, marker = m, c = c, label = label)
fig1.tight_layout()
fig1.subplots_adjust(right=0.9)
ax1.legend(loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=12)
ax1.view_init(10, 45)
ax1.set_xlabel('Power [kW]')
ax1.set_ylabel('Capacity [kWh]')
ax1.set_zlabel('Fittness LCOS [EURO/MWh]', labelpad=15)
ax1.set_title("Case 2 FF: Optimal LCOS convergence vs Iterations")



plt.savefig('Results\Pictures_etc\Compiled Results\Case-2-FF\Case2-FF-Optimal-LCOS-vs-iterations.jpeg',  dpi=300, bbox_inches = "tight")
plt.show()


#-------CASE 2 GA LCOS 3D SCATTER PLOT of Convergence of solutions

iter_ALL_combined = []
for count, i in enumerate(iterations):
    Data_GA_case_2_LCOS = pd.read_csv(f'Results\Pygad_case_2_ESS_LCOS\ESS_power_LCOS_etc\Pygad_case_2_LCOS_ESS_{i}_gen.csv')
    Power = Data_GA_case_2_LCOS["ESS_power"].values
    Capacity = Data_GA_case_2_LCOS["ESS_capacity"].values
    LCOS = (Data_GA_case_2_LCOS["fitness_function"].values)*1000
    combined = [LCOS, Capacity, Power]
    iter_ALL_combined.append(combined)


All = [(iter_ALL_combined[0],'^', 'r', '5 gen'),(iter_ALL_combined[1],'p', 'lawngreen', '10 gen'),
(iter_ALL_combined[2], 'X', 'g', '25 gen'), (iter_ALL_combined[3], 'D', 'b', '50 gen'),
(iter_ALL_combined[4], '1', 'coral', '100 gen'), (iter_ALL_combined[5], '+', 'olive', '200 gen')]


fig2 = plt.figure(3)
ax2 = fig2.add_subplot(projection='3d')
for t, m, c, label in All:
    xs = t[2]
    ys = t[1]
    zs = t[0]
    ax2.scatter(xs, ys, zs, marker = m, c = c, label = label)
fig2.tight_layout()
fig2.subplots_adjust(right=0.9)
ax2.legend(loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=12)
ax2.view_init(5, 30)
ax2.set_xlabel('Power [kW]')
ax2.set_ylabel('Capacity [kWh]')
ax2.set_zlabel('Fittness LCOS [EURO/MWh]', labelpad=15)
ax2.set_title("Case 2 GA: Optimal LCOS convergence vs Iterations")
ax2.axes.set_xlim3d(left=0, right=600) 
ax2.axes.set_ylim3d(bottom=100, top=1200) 


plt.savefig('Results\Pictures_etc\Compiled Results\Case-2-GA\Case2-GA-Optimal-LCOS-vs-iterations.jpeg',  dpi=300, bbox_inches = "tight")
plt.show()

#-------CASE 2 GA NPV 3D SCATTER PLOT of Convergence of solutions

iter_ALL_combined = []
for count, i in enumerate(iterations):
    Data_GA_case_2_NPV = pd.read_csv(f'Results\Pygad_case_2_ESS_NPV\ESS_power_NPV_etc\Pygad_case_2_NPV_ESS_{i}_gen.csv')
    Power = Data_GA_case_2_NPV["ESS_power"].values
    Capacity = Data_GA_case_2_NPV["ESS_capacity"].values
    NPV = Data_GA_case_2_NPV["fitness_function"].values
    combined = [NPV, Capacity, Power]
    iter_ALL_combined.append(combined)


All = [(iter_ALL_combined[0],'^', 'r', '5 gen'),(iter_ALL_combined[1],'p', 'lawngreen', '10 gen'),
(iter_ALL_combined[2], 'X', 'g', '25 gen'), (iter_ALL_combined[3], 'D', 'b', '50 gen'),
(iter_ALL_combined[4], '1', 'coral', '100 gen'), (iter_ALL_combined[5], '+', 'olive', '200 gen')]


fig3 = plt.figure(3)
ax3 = fig3.add_subplot(projection='3d')
for t, m, c, label in All:
    xs = t[2]
    ys = t[1]
    zs = t[0]
    ax3.scatter(xs, ys, zs, marker = m, c = c, label = label)
fig3.tight_layout()
fig3.subplots_adjust(right=0.9)
ax3.legend(loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=12)
ax3.view_init(20, 60)
ax3.set_xlabel('Power [kW]')
ax3.set_ylabel('Capacity [kWh]')
ax3.set_zlabel('Fittness NPV [EURO]', labelpad=15)
ax3.set_title("Case 2 GA: Optimal NPV convergence vs Iterations")

plt.savefig('Results\Pictures_etc\Compiled Results\Case-2-GA\Case2-GA-Optimal-NPV-vs-iterations.jpeg',  dpi=300, bbox_inches = "tight")
plt.show()

#------------Combined Case 2 NPV and LCOS for both GA and FF
fig4, ax4 = plt.subplots()

line1, = ax4.plot(iterations, Case2_GA_NPV,  label="GA NPV", linestyle='dashed', color = "red")
line2, = ax4.plot(iterations, Case2_FF_NPV, label="FF NPV", linestyle='dotted', color = "blue")

ax4.set_xlabel('Iterations')
ax4.set_ylabel('NPV [EURO]', color='blue')

ax4_2 = ax4.twinx()
ax4_2.set_ylabel("LCOS [Euro/MWh]", color = 'red')
line2, = ax4_2.plot(iterations, Case2_FF_LCOS*1000, label="FF LCOS", linewidth=1, color='lawngreen')
line2, = ax4_2.plot(iterations, Case2_GA_LCOS*1000, label="GA LCOS", linestyle ='dashdot', color= 'black')

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
ax5_2.set_ylabel("LCOS [Euro/MWh]", color = 'red')
line2, = ax5_2.plot(iterations, Case3_FF_LCOS*1000, label="FF LCOS", linewidth=1, color='lawngreen')
line2, = ax5_2.plot(iterations, Case3_GA_LCOS*1000, label="GA LCOS", linestyle ='dashdot',  color= 'black')

ax5.legend(loc='right', bbox_to_anchor=(1.4, 1))
ax5_2.legend(loc='right', bbox_to_anchor=(1.4, 0.8))
plt.title("Case 3 Average NPV and LCOS vs Iterations")
plt.savefig('Results\Pictures_etc\Compiled Results\Case-3-combined\Case3-NPV-LCOS-Combined-GA-FF.jpeg', dpi=300, bbox_inches = "tight")
plt.show
