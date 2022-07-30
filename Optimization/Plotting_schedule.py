
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('Results\Pygad_case_2_ESS_NPV\Charge_discharge_capacity\Pygad_case_2_NPV_ESS_200_gen_Sch_year_10.csv')
data_lcos = pd.read_csv("Results\Pygad_case_2_ESS_LCOS\Charge_discharge_capacity\Pygad_case_2_LCOS_ESS_200_gen_Sch_year_10.csv")

data_case3_npv = pd.read_csv("Results\Pygad_case_3_ESS_NPV\Charge_discharge_capacity\Pygad_case_3_ESS_NPV_200_gen_Sch_year_10.csv")
data_case3_lcos = pd.read_csv("Results\Pygad_case_3_ESS_LCOS\Charge_discharge_capacity\Pygad_case_3_ESS_LCOS_200_gen_Sch_year_10.csv")

charged_hours_npv = 0
discharge_hours_npv = 0
charged_hours_lcos = 0
discharge_hours_lcos = 0

case_3_charged_hours_npv = 0
case_3_discharge_hours_npv = 0
case_3_charged_hours_lcos = 0
case_3_discharge_hours_lcos = 0

for i in data["charge"]:
    if i > 0:
        charged_hours_npv += 1
for i in data["discharge"]:
    if i > 0:
        discharge_hours_npv += 1


for i in data_lcos["charge"]:
    if i > 0:
        charged_hours_lcos += 1
for i in data_lcos["discharge"]:
    if i > 0:
        discharge_hours_lcos += 1

for i in data_case3_npv["charge"]:
    if i > 0:
        case_3_charged_hours_npv += 1
for i in data_case3_npv["discharge"]:
    if i > 0:
        case_3_discharge_hours_npv += 1


for i in data_case3_lcos["charge"]:
    if i > 0:
        case_3_charged_hours_lcos += 1
for i in data_case3_lcos["discharge"]:
    if i > 0:
        case_3_discharge_hours_lcos += 1


Case_2_NPV = [charged_hours_npv, discharge_hours_npv, (8760 - charged_hours_npv - discharge_hours_npv)]
Case_2_LCOS = [charged_hours_lcos, discharge_hours_lcos, (8760 - charged_hours_lcos - discharge_hours_lcos)]

Case_3_NPV = [case_3_charged_hours_npv, case_3_discharge_hours_npv, (8760 - case_3_charged_hours_npv - case_3_discharge_hours_npv)]
Case_3_LCOS = [case_3_charged_hours_lcos, case_3_discharge_hours_lcos, (8760 - case_3_charged_hours_lcos - case_3_discharge_hours_lcos)]

explode = (0.1, 0.1, 0)
labels = ["Charge", "Discharge", "Passive"]
fig, ax = plt.subplots(2,2)

ax[0,0].pie(Case_2_NPV, startangle=90, labels=labels, explode=explode, autopct="%1.1f%%", labeldistance = None, pctdistance = 0.8, textprops = {'size': 'x-small'})
ax[0,0].axis('equal')
ax[0,0].set_title("Case 2: NPV [11.3 kWh, 5.6 kW] ", pad = 10.0, fontsize = 8)

ax[0,1].pie(Case_2_LCOS, startangle=90, explode=explode, autopct="%1.1f%%", labeldistance = None, textprops = {'size': 'x-small'})
ax[0,1].axis('equal')
ax[0,1].set_title("Case 2: LCOS [607.9 kWh, 181.2 kW]", pad = 10.0, fontsize = 8)

ax[1,0].pie(Case_3_NPV, startangle=90, explode=explode, autopct="%1.1f%%", labeldistance = None, pctdistance = 0.8, textprops = {'size': 'x-small'})
ax[1,0].axis('equal')
ax[1,0].set_title("Case 3: NPV [3.8 kWh, 11.9 kW]", pad = 10.0, fontsize = 8)

ax[1,1].pie(Case_3_LCOS, startangle=90, explode=explode, autopct="%1.1f%%", labeldistance = None, textprops = {'size': 'x-small'})
ax[1,1].axis('equal')
ax[1,1].set_title("Case 3: LCOS [625.3 kWh, 188.3 kW]", pad = 10.0, fontsize = 8)



fig.legend(bbox_to_anchor=(1.2, 1), fontsize=12, title = "[Capacity, Power]")
plt.suptitle("Pygad 200 iterations Schedule", y = 1.025)


plt.savefig('Results\Pictures_etc\Schedule\Pie_chart_schedule_pygad.jpeg',  dpi=300, bbox_inches = "tight")

plt.show()