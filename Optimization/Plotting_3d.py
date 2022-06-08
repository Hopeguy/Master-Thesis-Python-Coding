
from ast import Load
from turtle import color
from matplotlib import projections

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
iterations = ['5', '10', '25', '50', '100', '200']
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


fig = plt.figure(0)
ax = fig.add_subplot(projection='3d')
for t, m, c, label in All:
    xs = t[2]
    ys = t[1]
    zs = t[0]
    ax.scatter(xs, ys, zs, marker = m, c = c, label = label)
fig.tight_layout()
fig.subplots_adjust(right=0.9)
ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=12)
ax.view_init(20, 45)
ax.set_xlabel('Power [kW]')
ax.set_ylabel('Capacity [kWh]')
ax.set_zlabel('NPV [EURO]', labelpad=15)
ax.set_title("Case 2 FF: Optimal NPV convergence vs Iterations")



plt.savefig('Results\Pictures_etc\Compiled Results\Case-2-FF\Case2-FF-Optimal-NPV-vs-iterations.jpeg',  dpi=300, bbox_inches = "tight")
plt.show()




#--------CASE 2 FF LCOS 3D SCATTER PLOT of Convergence of solutions

iter_ALL_combined = []
for count, i in enumerate(iterations):
    Data_FF_case_2_LCOS = pd.read_csv(f'Results\Firefly_case_2_ESS_LCOS\ESS_power_LCOS_etc\Firefly_case_2_ESS_LCOS_{i}_gen.csv')
    Power = Data_FF_case_2_LCOS["ESS_power"].values
    Capacity = Data_FF_case_2_LCOS["ESS_capacity"].values
    LCOS = (Data_FF_case_2_LCOS["fitness_function"].values)
    combined = [LCOS, Capacity, Power]
    iter_ALL_combined.append(combined)


All = [(iter_ALL_combined[0],'^', 'r', '5 gen'),(iter_ALL_combined[1],'p', 'lawngreen', '10 gen'),
(iter_ALL_combined[2], 'X', 'g', '25 gen'), (iter_ALL_combined[3], 'D', 'b', '50 gen'),
(iter_ALL_combined[4], '1', 'coral', '100 gen'), (iter_ALL_combined[5], '+', 'olive', '200 gen')]


fig1 = plt.figure(1)
ax1 = fig1.add_subplot(projection='3d')
for t, m, c, label in All:
    xs = t[2]
    ys = t[1]
    zs = t[0]
    ax1.scatter(xs, ys, zs, marker = m, c = c, label = label)
fig1.tight_layout()
fig1.subplots_adjust(right=0.9)
ax1.legend(loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=12)
ax1.view_init(10, 30)
ax1.set_xlabel('Power [kW]')
ax1.set_ylabel('Capacity [kWh]')
ax1.set_zlabel('Fittness LCOS [EURO/kWh]', labelpad=15)
ax1.set_title("Case 2 FF: Optimal LCOS convergence vs Iterations")



plt.savefig('Results\Pictures_etc\Compiled Results\Case-2-FF\Case2-FF-Optimal-LCOS-vs-iterations.jpeg',  dpi=300, bbox_inches = "tight")
plt.show()


#-------CASE 2 GA LCOS 3D SCATTER PLOT of Convergence of solutions

iter_ALL_combined = []
for count, i in enumerate(iterations):
    Data_GA_case_2_LCOS = pd.read_csv(f'Results\Pygad_case_2_ESS_LCOS\ESS_power_LCOS_etc\Pygad_case_2_LCOS_ESS_{i}_gen.csv')
    Power = Data_GA_case_2_LCOS["ESS_power"].values
    Capacity = Data_GA_case_2_LCOS["ESS_capacity"].values
    LCOS = (Data_GA_case_2_LCOS["fitness_function"].values)
    combined = [LCOS, Capacity, Power]
    iter_ALL_combined.append(combined)


All = [(iter_ALL_combined[0],'^', 'r', '5 gen'),(iter_ALL_combined[1],'p', 'lawngreen', '10 gen'),
(iter_ALL_combined[2], 'X', 'g', '25 gen'), (iter_ALL_combined[3], 'D', 'b', '50 gen'),
(iter_ALL_combined[4], '1', 'coral', '100 gen'), (iter_ALL_combined[5], '+', 'olive', '200 gen')]


fig2 = plt.figure(2)
ax2 = fig2.add_subplot(projection='3d')
for t, m, c, label in All:
    xs = t[2]
    ys = t[1]
    zs = t[0]
    ax2.scatter(xs, ys, zs, marker = m, c = c, label = label)
fig2.tight_layout()
fig2.subplots_adjust(right=0.9)
ax2.legend(loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=12)
ax2.view_init(15, 20)
ax2.set_xlabel('Power [kW]')
ax2.set_ylabel('Capacity [kWh]')
ax2.set_zlabel('Fittness LCOS [EURO/MWh]', labelpad=15)
ax2.set_title("Case 2 GA: Optimal LCOS convergence vs Iterations")
ax2.xaxis.set_tick_params(labelsize =9)
ax2.yaxis.set_tick_params(labelsize =9)
ax2.axes.set_xlim3d(left=0, right=600) 
ax2.axes.set_ylim3d(bottom=100, top=1400) 


plt.savefig('Results\Pictures_etc\Compiled Results\Case-2-GA\Case2-GA-Optimal-LCOS-vs-iterations.jpeg',  dpi=300, bbox_inches = "tight")
plt.show()

#-------CASE 2 GA NPV 3D SCATTER PLOT of Convergence of solutions

iter_ALL_combined = []
iter_ALL_combined_2D = []
for count, i in enumerate(iterations):
    Data_GA_case_2_NPV = pd.read_csv(f'Results\Pygad_case_2_ESS_NPV\ESS_power_NPV_etc\Pygad_case_2_NPV_ESS_{i}_gen.csv')
    Power = Data_GA_case_2_NPV["ESS_power"].values
    Capacity = Data_GA_case_2_NPV["ESS_capacity"].values
    NPV = Data_GA_case_2_NPV["fitness_function"].values
    combined = [NPV, Capacity, Power]
    iter_ALL_combined_2D.append([Capacity, Power])
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
ax3.view_init(20, 15)
ax3.set_xlabel('Power [kW]')
ax3.set_ylabel('Capacity [kWh]')
ax3.set_zlabel('Fittness NPV [EURO]', labelpad=15)
ax3.set_title("Case 2 GA: Optimal NPV convergence vs Iterations")
ax3.zaxis.set_tick_params(labelsize = 8)

plt.savefig('Results\Pictures_etc\Compiled Results\Case-2-GA\Case2-GA-Optimal-NPV-vs-iterations.jpeg',  dpi=300, bbox_inches = "tight")
plt.show()



All_2D = [(iter_ALL_combined_2D[0],'^', 'r', '5 gen'),(iter_ALL_combined_2D[1],'p', 'lawngreen', '10 gen'),
(iter_ALL_combined_2D[2], 'X', 'g', '25 gen'), (iter_ALL_combined_2D[3], 'D', 'b', '50 gen'),
(iter_ALL_combined_2D[4], '1', 'coral', '100 gen'), (iter_ALL_combined_2D[5], '+', 'olive', '200 gen')]


fig4, ax4 = plt.subplots()
for t, m, c, label in All_2D:
    xs = t[1]
    ys = t[0]  
    plt.scatter(x = xs, y = ys, marker = m, c = c, label = label)

ax4.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
ax4.set_ylabel("Capacity [kWh]")
ax4.set_xlabel('Power [kW]')
ax4.grid()
plt.title("Case 2 GA NPV, Power and capacity vs Iterations")

plt.savefig('Results\Pictures_etc\Compiled Results\Case-2-GA\Case2-GA-Optimal-Capcaity-Power-vs-iterations.jpeg',  dpi=300, bbox_inches = "tight")
plt.show()

## -------------3D CASE 3 ELH POWER, BESS POWER and CAPACITY

iter_ALL_combined = []
for count, i in enumerate(iterations):
    Data_FF_case_3_NPV = pd.read_csv(f'Results\Firefly_case_3_ESS_NPV\ESS_power_NPV_etc\Firefly_case_3_ESS_{i}_gen.csv')
    Power = Data_FF_case_3_NPV["ESS_power"].values
    Capacity = Data_FF_case_3_NPV["ESS_capacity"].values
    ELH_power = (Data_FF_case_3_NPV["ELH_power"].values)
    combined = [ELH_power, Capacity, Power]
    iter_ALL_combined.append(combined)


All = [(iter_ALL_combined[0],'^', 'r', '5 gen'),(iter_ALL_combined[1],'p', 'lawngreen', '10 gen'),
(iter_ALL_combined[2], 'X', 'g', '25 gen'), (iter_ALL_combined[3], 'D', 'b', '50 gen'),
(iter_ALL_combined[4], '1', 'coral', '100 gen'), (iter_ALL_combined[5], '+', 'olive', '200 gen')]


fig5 = plt.figure(2)
ax5 = fig5.add_subplot(projection='3d')
for t, m, c, label in All:
    xs = t[2]
    ys = t[1]
    zs = t[0]
    ax5.scatter(xs, ys, zs, marker = m, c = c, label = label)
fig5.tight_layout()
fig5.subplots_adjust(right=0.9)
ax5.legend(loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=12)
ax5.view_init(30, 20)
ax5.set_xlabel('Power [kW]')
ax5.set_ylabel('Capacity [kWh]')
ax5.set_zlabel('ELH Power [kW]]', labelpad=15)
ax5.set_title("Case 3 FF: Optimal NPV convergence vs Iterations")
ax5.xaxis.set_tick_params(labelsize =9)
ax5.yaxis.set_tick_params(labelsize =9)
ax5.axes.set_xlim3d(left=0, right=1400) 
ax5.axes.set_ylim3d(bottom=0, top=950) 

plt.savefig('Results\Pictures_etc\Compiled Results\Case-3-FF\Case3-FF-Optimal-Capcaity-Power-ELH-Power-NPV-vs-iterations.jpeg',  dpi=300, bbox_inches = "tight")
plt.show()





iter_ALL_combined = []
for count, i in enumerate(iterations):
    Data_GA_case_3_NPV = pd.read_csv(f'Results\Pygad_case_3_ESS_NPV\ESS_power_NPV_etc\Pygad_case_3_ESS_NPV_{i}_gen.csv')
    Power = Data_GA_case_3_NPV["ESS_power"].values
    Capacity = Data_GA_case_3_NPV["ESS_capacity"].values
    ELH_power = (Data_GA_case_3_NPV["ELH_power"].values)
    combined = [ELH_power, Capacity, Power]
    iter_ALL_combined.append(combined)


All = [(iter_ALL_combined[0],'^', 'r', '5 gen'),(iter_ALL_combined[1],'p', 'lawngreen', '10 gen'),
(iter_ALL_combined[2], 'X', 'g', '25 gen'), (iter_ALL_combined[3], 'D', 'b', '50 gen'),
(iter_ALL_combined[4], '1', 'coral', '100 gen'), (iter_ALL_combined[5], '+', 'olive', '200 gen')]


fig6 = plt.figure(2)
ax6 = fig6.add_subplot(projection='3d')
for t, m, c, label in All:
    xs = t[2]
    ys = t[1]
    zs = t[0]
    ax6.scatter(xs, ys, zs, marker = m, c = c, label = label)
fig6.tight_layout()
fig6.subplots_adjust(right=0.9)
ax6.legend(loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=12)
ax6.view_init(30, 60)
ax6.set_xlabel('Power [kW]')
ax6.set_ylabel('Capacity [kWh]')
ax6.set_zlabel('ELH Power [kW]]', labelpad=15)
ax6.set_title("Case 3 GA: Optimal NPV convergence vs Iterations")
ax6.xaxis.set_tick_params(labelsize =9)
ax6.yaxis.set_tick_params(labelsize =9)
ax6.axes.set_xlim3d(left=0, right=400) 
ax6.axes.set_ylim3d(bottom=0, top=550) 

plt.savefig('Results\Pictures_etc\Compiled Results\Case-3-GA\Case3-GA-Optimal-Capcaity-Power-ELH-Power-NPV-vs-iterations.jpeg',  dpi=300, bbox_inches = "tight")
plt.show()




iter_ALL_combined = []
for count, i in enumerate(iterations):
    Data_GA_case_3_LCOS = pd.read_csv(f'Results\Pygad_case_3_ESS_LCOS\ESS_power_LCOS_etc\Pygad_case_3_ESS_LCOS_{i}_gen.csv')
    Power = Data_GA_case_3_LCOS["ESS_power"].values
    Capacity = Data_GA_case_3_LCOS["ESS_capacity"].values
    ELH_power = (Data_GA_case_3_LCOS["ELH_power"].values)
    combined = [ELH_power, Capacity, Power]
    iter_ALL_combined.append(combined)


All = [(iter_ALL_combined[0],'^', 'r', '5 gen'),(iter_ALL_combined[1],'p', 'lawngreen', '10 gen'),
(iter_ALL_combined[2], 'X', 'g', '25 gen'), (iter_ALL_combined[3], 'D', 'b', '50 gen'),
(iter_ALL_combined[4], '1', 'coral', '100 gen'), (iter_ALL_combined[5], '+', 'olive', '200 gen')]


fig7 = plt.figure()
ax7 = fig7.add_subplot(projection='3d')
for t, m, c, label in All:
    xs = t[2]
    ys = t[1]
    zs = t[0]
    ax7.scatter(xs, ys, zs, marker = m, c = c, label = label)
fig7.tight_layout()
fig7.subplots_adjust(right=0.9)
ax7.legend(loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=12)
ax7.view_init(20, 75)
ax7.set_xlabel('Power [kW]')
ax7.set_ylabel('Capacity [kWh]')
ax7.set_zlabel('ELH Power [kW]]', labelpad=15)
ax7.set_title("Case 3 GA: Optimal LCOS convergence vs Iterations")
ax7.xaxis.set_tick_params(labelsize =9)
ax7.yaxis.set_tick_params(labelsize =9)


plt.savefig('Results\Pictures_etc\Compiled Results\Case-3-GA\Case3-GA-Optimal-Capcaity-Power-ELH-Power-LCOS-vs-iterations.jpeg',  dpi=300, bbox_inches = "tight")
plt.show()


iter_ALL_combined = []
for count, i in enumerate(iterations):
    Data_FF_case_3_LCOS = pd.read_csv(f'Results\Firefly_case_3_ESS_LCOS\ESS_power_LCOS_etc\Firefly_case_3_ESS_LCOS_{i}_gen.csv')
    Power = Data_FF_case_3_LCOS["ESS_power"].values
    Capacity = Data_FF_case_3_LCOS["ESS_capacity"].values
    ELH_power = (Data_FF_case_3_LCOS["ELH_power"].values)
    combined = [ELH_power, Capacity, Power]
    iter_ALL_combined.append(combined)


All = [(iter_ALL_combined[0],'^', 'r', '5 gen'),(iter_ALL_combined[1],'p', 'lawngreen', '10 gen'),
(iter_ALL_combined[2], 'X', 'g', '25 gen'), (iter_ALL_combined[3], 'D', 'b', '50 gen'),
(iter_ALL_combined[4], '1', 'coral', '100 gen'), (iter_ALL_combined[5], '+', 'olive', '200 gen')]


fig8 = plt.figure()
ax8 = fig8.add_subplot(projection='3d')
for t, m, c, label in All:
    xs = t[2]
    ys = t[1]
    zs = t[0]
    ax8.scatter(xs, ys, zs, marker = m, c = c, label = label)
fig8.tight_layout()
fig8.subplots_adjust(right=0.9)
ax8.legend(loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=12)
ax8.view_init(20, 50)
ax8.set_xlabel('Power [kW]')
ax8.set_ylabel('Capacity [kWh]')
ax8.set_zlabel('ELH Power [kW]]', labelpad=15)
ax8.set_title("Case 3 FF: Optimal LCOS convergence vs Iterations")
ax8.xaxis.set_tick_params(labelsize =9)
ax8.yaxis.set_tick_params(labelsize =9)

plt.savefig('Results\Pictures_etc\Compiled Results\Case-3-FF\Case3-FF-Optimal-Capcaity-Power-ELH-Power-LCOS-vs-iterations.jpeg',  dpi=300, bbox_inches = "tight")
plt.show()