from ast import Load
from turtle import color
from matplotlib import projections

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iterations =['5', '10', '25', '50', '100', '200']
iterations_num = [5, 10, 25, 50, 100, 200]
data = pd.read_csv('Results/Results_for_printing.csv')

Case2_GA_NPV_STD, Case2_GA_LCOS_STD, Case2_FF_NPV_STD, Case2_FF_LCOS_STD = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
Case3_GA_NPV_STD, Case3_GA_LCOS_STD, Case3_FF_NPV_STD, Case3_FF_LCOS_STD = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
Case2_GA_NPV_TIME_STD, Case2_GA_LCOS_TIME_STD, Case2_FF_NPV_TIME_STD, Case2_FF_LCOS_TIME_STD = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
Case3_GA_NPV_TIME_STD, Case3_GA_LCOS_TIME_STD, Case3_FF_NPV_TIME_STD, Case3_FF_LCOS_TIME_STD = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
Case2_GA_NPV_TIME, Case2_GA_LCOS_TIME, Case2_FF_NPV_TIME, Case2_FF_LCOS_TIME = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
Case3_GA_NPV_TIME, Case3_GA_LCOS_TIME, Case3_FF_NPV_TIME, Case3_FF_LCOS_TIME = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
Case2_GA_NPV_AVG, Case2_GA_LCOS_AVG, Case2_FF_NPV_AVG, Case2_FF_LCOS_AVG = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
Case3_GA_NPV_AVG, Case3_GA_LCOS_AVG, Case3_FF_NPV_AVG, Case3_FF_LCOS_AVG = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)


for count, i in enumerate(iterations):
    #CASE 2 STD LCOS AND NPV
    Case2_GA_NPV_STD[count] = data.iloc[36][f'{i}']
    Case2_GA_LCOS_STD[count] = data.iloc[37][f'{i}']
    Case2_FF_NPV_STD[count] = data.iloc[38][f'{i}']
    Case2_FF_LCOS_STD[count] = data.iloc[39][f'{i}']

    #CASE 3 STD LCOS AND 
    
    Case3_GA_NPV_STD[count] = data.iloc[40][f'{i}']
    Case3_GA_LCOS_STD[count] = data.iloc[41][f'{i}']
    Case3_FF_NPV_STD[count] = data.iloc[42][f'{i}']
    Case3_FF_LCOS_STD[count] = data.iloc[43][f'{i}']

    #For time CASE 2
    Case2_GA_NPV_TIME[count] = data.iloc[3][f'{i}']
    Case2_GA_LCOS_TIME[count] = data.iloc[7][f'{i}']
    Case2_FF_NPV_TIME[count] = data.iloc[11][f'{i}']
    Case2_FF_LCOS_TIME[count] = data.iloc[15][f'{i}']

    #For time Case 3
    Case3_GA_NPV_TIME[count] = data.iloc[20][f'{i}']
    Case3_GA_LCOS_TIME[count] = data.iloc[25][f'{i}']
    Case3_FF_NPV_TIME[count] = data.iloc[30][f'{i}']
    Case3_FF_LCOS_TIME[count] = data.iloc[35][f'{i}']

    #For time STD CASE 2
    Case2_GA_NPV_TIME_STD[count] = data.iloc[44][f'{i}']
    Case2_GA_LCOS_TIME_STD[count] = data.iloc[45][f'{i}']
    Case2_FF_NPV_TIME_STD[count] = data.iloc[46][f'{i}']
    Case2_FF_LCOS_TIME_STD[count] = data.iloc[47][f'{i}']

    #For time STD Case 3
    Case3_GA_NPV_TIME_STD[count] = data.iloc[48][f'{i}']
    Case3_GA_LCOS_TIME_STD[count] = data.iloc[49][f'{i}']
    Case3_FF_NPV_TIME_STD[count] = data.iloc[50][f'{i}']
    Case3_FF_LCOS_TIME_STD[count] = data.iloc[51][f'{i}']

    #For average value for Case 2

    Case2_GA_NPV_AVG[count] = data.iloc[0][f'{i}']
    Case2_GA_LCOS_AVG[count] = data.iloc[4][f'{i}']
    Case2_FF_NPV_AVG[count] = data.iloc[8][f'{i}']
    Case2_FF_LCOS_AVG[count] = data.iloc[12][f'{i}']

    # For average value for case 3
    Case3_GA_NPV_AVG[count] = data.iloc[16][f'{i}']
    Case3_GA_LCOS_AVG[count] = data.iloc[21][f'{i}']
    Case3_FF_NPV_AVG[count] = data.iloc[26][f'{i}']
    Case3_FF_LCOS_AVG[count] = data.iloc[31][f'{i}']




fig, ax = plt.subplots()

line1, = ax.plot(iterations, Case2_GA_NPV_STD,  label="Case 2 GA NPV", linestyle='dashed', color = "red")
line2, = ax.plot(iterations, Case2_FF_NPV_STD, label="Case 2 FF NPV", linestyle='dotted', color = "cyan")
line3, = ax.plot(iterations, Case3_GA_NPV_STD, label = "Case 3 GA NPV", linewidth = 1, color = 'lawngreen')
line4, = ax.plot(iterations, Case3_FF_NPV_STD, label = "Case 3 FF NPV", linestyle ='dashdot', color = 'black')

ax.set_xlabel('Iterations')
ax.set_ylabel('NPV [EURO]', color='blue')
ax.grid()

ax_2 = ax.twinx()
ax_2.set_ylabel("LCOS [Euro/MWh]", color = 'red')
line5, = ax_2.plot(iterations, Case2_GA_LCOS_STD, label="Case 2 GA LCOS", linestyle = 'dashed', color='salmon')
line6, = ax_2.plot(iterations, Case2_FF_LCOS_STD, label="Case 2 FF LCOS", linestyle ='dotted', color= 'navy')
line7, = ax_2.plot(iterations, Case3_GA_LCOS_STD, label="Case 3 GA LCOS", linewidth = 1, color= 'lime')
line8, = ax_2.plot(iterations, Case3_FF_LCOS_STD, label="Case 3 FF LCOS", linestyle ='dashdot', color= 'grey')


ax.legend(loc='right', bbox_to_anchor=(1.5, 1))
ax_2.legend(loc='right', bbox_to_anchor=(1.52, 0.68))
plt.title("STD NPV and LCOS vs Iterations")
plt.savefig('Results\Pictures_etc\Compiled Results\STD-AlL-CASES-LCOS-NPV.jpeg', dpi=300, bbox_inches = "tight")
plt.show


fig2, ax2 = plt.subplots()

ax2.set_xlabel('Iterations')
ax2.set_ylabel('Time [Seconds]', color='black')
ax2.grid()

line1, = ax2.plot(iterations, Case2_GA_NPV_TIME,  label="Case 2 GA NPV", linestyle='dashed', color = "red")
line2, = ax2.plot(iterations, Case2_FF_NPV_TIME, label="Case 2 FF NPV", linestyle='dotted', color = "cyan")
line3, = ax2.plot(iterations, Case3_GA_NPV_TIME, label = "Case 3 GA NPV", linewidth = 1, color = 'lawngreen')
line4, = ax2.plot(iterations, Case3_FF_NPV_TIME, label = "Case 3 FF NPV", linestyle ='dashdot', color = 'black')

line5, = ax2.plot(iterations, Case2_GA_LCOS_TIME, label="Case 2 GA LCOS", linestyle = 'dashed', color='salmon')
line6, = ax2.plot(iterations, Case2_FF_LCOS_TIME, label="Case 2 FF LCOS", linestyle ='dotted', color= 'navy')
line7, = ax2.plot(iterations, Case3_GA_LCOS_TIME, label="Case 3 GA LCOS", linewidth = 1, color= 'lime')
line8, = ax2.plot(iterations, Case3_FF_LCOS_TIME, label="Case 3 FF LCOS", linestyle ='dashdot', color= 'grey')
plt.title("Time NPV and LCOS vs Iterations")
ax2.legend(loc='right', bbox_to_anchor=(1.4, 0.72))

plt.savefig('Results\Pictures_etc\Compiled Results\TIME-AlL-CASES-LCOS-NPV.jpeg', dpi=300, bbox_inches = "tight")
plt.show


fig3, ax3 = plt.subplots()

ax3.set_xlabel('Iterations')
ax3.set_ylabel('STD Time [Seconds]', color='black')
ax3.grid()
plt.yscale("log")

line1, = ax3.plot(iterations, Case2_GA_NPV_TIME_STD,  label="Case 2 GA NPV", linestyle='dashed', color = "red")
line2, = ax3.plot(iterations, Case2_FF_NPV_TIME_STD, label="Case 2 FF NPV", linestyle='dotted', color = "cyan")
line3, = ax3.plot(iterations, Case3_GA_NPV_TIME_STD, label = "Case 3 GA NPV", linewidth = 1, color = 'lawngreen')
line4, = ax3.plot(iterations, Case3_FF_NPV_TIME_STD, label = "Case 3 FF NPV", linestyle ='dashdot', color = 'black')

line5, = ax3.plot(iterations, Case2_GA_LCOS_TIME_STD, label="Case 2 GA LCOS", linestyle = 'dashed', color='salmon')
line6, = ax3.plot(iterations, Case2_FF_LCOS_TIME_STD, label="Case 2 FF LCOS", linestyle ='dotted', color= 'navy')
line7, = ax3.plot(iterations, Case3_GA_LCOS_TIME_STD, label="Case 3 GA LCOS", linewidth = 1, color= 'lime')
line8, = ax3.plot(iterations, Case3_FF_LCOS_TIME_STD, label="Case 3 FF LCOS", linestyle ='dashdot', color= 'grey')
plt.title("STD Time NPV and LCOS vs Iterations")
ax3.legend(loc='right', bbox_to_anchor=(1.4, 0.72))

plt.savefig('Results\Pictures_etc\Compiled Results\TIME-STD-AlL-CASES-LCOS-NPV.jpeg', dpi=300, bbox_inches = "tight")
plt.show


fig4, (ax4, ax5) = plt.subplots(1,2, figsize=(9, 4))
fig4.suptitle("Time and STD for NPV and LCOS vs Iterations")

ax4.grid()
ax4.set_xlabel('Iterations')
ax4.set_ylabel('Time [Seconds]', color='black')

line1, = ax4.plot(iterations, Case2_GA_NPV_TIME,  label="Case 2 GA NPV", linestyle='dashed', color = "red")
line2, = ax4.plot(iterations, Case2_FF_NPV_TIME, label="Case 2 FF NPV", linestyle='dotted', color = "cyan")
line3, = ax4.plot(iterations, Case3_GA_NPV_TIME, label = "Case 3 GA NPV", linewidth = 1, color = 'lawngreen')
line4, = ax4.plot(iterations, Case3_FF_NPV_TIME, label = "Case 3 FF NPV", linestyle ='dashdot', color = 'black')

line5, = ax4.plot(iterations, Case2_GA_LCOS_TIME, label="Case 2 GA LCOS", linestyle = 'dashed', color='salmon')
line6, = ax4.plot(iterations, Case2_FF_LCOS_TIME, label="Case 2 FF LCOS", linestyle ='dotted', color= 'navy')
line7, = ax4.plot(iterations, Case3_GA_LCOS_TIME, label="Case 3 GA LCOS", linewidth = 1, color= 'lime')
line8, = ax4.plot(iterations, Case3_FF_LCOS_TIME, label="Case 3 FF LCOS", linestyle ='dashdot', color= 'grey')

ax5.set_xlabel('Iterations')
ax5.set_ylabel('STD Time [Seconds]', color='black')
ax5.grid()
ax5.set_yscale("log")

line1, = ax5.plot(iterations, Case2_GA_NPV_TIME_STD,  label="Case 2 GA NPV", linestyle='dashed', color = "red")
line2, = ax5.plot(iterations, Case2_FF_NPV_TIME_STD, label="Case 2 FF NPV", linestyle='dotted', color = "cyan")
line3, = ax5.plot(iterations, Case3_GA_NPV_TIME_STD, label = "Case 3 GA NPV", linewidth = 1, color = 'lawngreen')
line4, = ax5.plot(iterations, Case3_FF_NPV_TIME_STD, label = "Case 3 FF NPV", linestyle ='dashdot', color = 'black')

line5, = ax5.plot(iterations, Case2_GA_LCOS_TIME_STD, label="Case 2 GA LCOS", linestyle = 'dashed', color='salmon')
line6, = ax5.plot(iterations, Case2_FF_LCOS_TIME_STD, label="Case 2 FF LCOS", linestyle ='dotted', color= 'navy')
line7, = ax5.plot(iterations, Case3_GA_LCOS_TIME_STD, label="Case 3 GA LCOS", linewidth = 1, color= 'lime')
line8, = ax5.plot(iterations, Case3_FF_LCOS_TIME_STD, label="Case 3 FF LCOS", linestyle ='dashdot', color= 'grey')

ax5.legend(loc='right', bbox_to_anchor=(1.6, 0.72))
plt.savefig('Results\Pictures_etc\Compiled Results\TIME-STD-AlL-CASES-LCOS-NPV-COMBINED.jpeg', dpi=300, bbox_inches = "tight")


#Calculate difference in optimization time

Case2_NPV = 1 - (sum(Case2_GA_NPV_TIME)/sum(Case2_FF_NPV_TIME))
Case2_LCOS = 1 - (sum(Case2_GA_LCOS_TIME)/sum(Case2_FF_LCOS_TIME))
Case3_NPV = 1 - (sum(Case3_GA_NPV_TIME)/sum(Case3_FF_NPV_TIME))
Case3_LCOS = 1 - (sum(Case3_GA_LCOS_TIME)/sum(Case3_FF_LCOS_TIME))

Time_diff = [Case2_NPV, Case2_LCOS, Case3_NPV, Case3_LCOS]

print(Time_diff)


fig6, ax6 = plt.subplots()

ax6.errorbar(iterations, Case2_GA_NPV_AVG, Case2_GA_NPV_STD, label = ' Case 2 GA NPV', linestyle='None', marker='^')
ax6.errorbar(iterations, Case2_FF_NPV_AVG, Case2_FF_NPV_STD, label = 'Case 2 FF NPV', linestyle='None', marker='x')
ax6.errorbar(iterations, Case3_GA_NPV_AVG, Case3_GA_NPV_STD, label = 'Case 3 GA NPV', linestyle='None', marker='p')
ax6.errorbar(iterations, Case3_FF_NPV_AVG, Case3_FF_NPV_STD, label = 'Case 3 FF NPV', linestyle='None', marker='o')

ax6.set_ylabel('NPV [EURO]')
ax6.set_xlabel('Iterations')
ax6.grid()
ax6.legend(loc='right', bbox_to_anchor=(1.4, 0.85))
ax6.set_title('Average NPV with STD')

plt.savefig('Results\Pictures_etc\Compiled Results\All_NPV-Average-and-STD.jpeg', dpi=300, bbox_inches = "tight")


fig7, ax7 = plt.subplots()

ax7.errorbar(iterations, Case2_GA_LCOS_AVG, Case2_GA_LCOS_STD, label = 'Case 2 GA LCOS', linestyle='None', marker='^')
ax7.errorbar(iterations, Case2_FF_LCOS_AVG, Case2_FF_LCOS_STD, label = 'Case 2 FF LCOS', linestyle='None', marker='x')
ax7.errorbar(iterations, Case3_GA_LCOS_AVG, Case3_GA_LCOS_STD, label = 'Case 3 GA LCOS', linestyle='None', marker='p')
ax7.errorbar(iterations, Case3_FF_LCOS_AVG, Case3_FF_LCOS_STD, label = 'Case 3 FF LCOS', linestyle='None', marker='o')

ax7.set_ylabel('LCOS [kWh/EURO]')
ax7.set_xlabel('Iterations')
ax7.grid()
ax7.legend(loc='right', bbox_to_anchor=(1.4, 0.85))
ax7.set_title('Average LCOS with STD')

plt.savefig('Results\Pictures_etc\Compiled Results\LCOS-Average-and-STD.jpeg', dpi=300, bbox_inches = "tight")



fig8, (ax8, ax9) = plt.subplots(1,2, figsize=(9, 4))
fig8.suptitle("Average NPV and LCOS wtih STD")

ax8.grid()
ax8.set_xlabel('Iterations')
ax8.set_ylabel('NPV [EURO]', color='black')


ax8.errorbar(iterations, Case2_GA_NPV_AVG, Case2_GA_NPV_STD, label = 'Case 2 GA NPV', linestyle='None', marker='^')
ax8.errorbar(iterations, Case2_FF_NPV_AVG, Case2_FF_NPV_STD, label = 'Case 2 FF NPV', linestyle='None', marker='x')
ax8.errorbar(iterations, Case3_GA_NPV_AVG, Case3_GA_NPV_STD, label = 'Case 3 GA NPV', linestyle='None', marker='p')
ax8.errorbar(iterations, Case3_FF_NPV_AVG, Case3_FF_NPV_STD, label = 'Case 3 FF NPV', linestyle='None', marker='o')

ax9.set_xlabel('Iterations')
ax9.set_ylabel('LCOS [EURO/kWh]', color='black')
ax9.grid()

ax9.errorbar(iterations, Case2_GA_LCOS_AVG, Case2_GA_LCOS_STD, label = 'Case 2 GA', linestyle='None', marker='^')
ax9.errorbar(iterations, Case2_FF_LCOS_AVG, Case2_FF_LCOS_STD, label = 'Case 2 FF', linestyle='None', marker='x')
ax9.errorbar(iterations, Case3_GA_LCOS_AVG, Case3_GA_LCOS_STD, label = 'Case 3 GA', linestyle='None', marker='p')
ax9.errorbar(iterations, Case3_FF_LCOS_AVG, Case3_FF_LCOS_STD, label = 'Case 3 FF', linestyle='None', marker='o')

ax9.legend(loc='right', bbox_to_anchor=(1.45, 0.85))


plt.savefig('Results\Pictures_etc\Compiled Results\LCOS-Average-and-STD_Combined.jpeg', dpi=300, bbox_inches = "tight")