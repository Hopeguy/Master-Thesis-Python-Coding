import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import std
from scipy.stats import norm

Case_3_NPV_GA = pd.read_csv('Results\Pygad_case_3_ESS_NPV\ESS_power_NPV_etc\Pygad_case_3_ESS_NPV_200_gen.csv')
Case_3_NPV_FF = pd.read_csv('Results\Firefly_case_3_ESS_NPV\ESS_power_NPV_etc\Firefly_case_3_ESS_200_gen.csv')

NPV_GA = abs(Case_3_NPV_GA['fitness_function'])
NPV_FF = abs(Case_3_NPV_FF['fitness_function'])

NPV_mean_GA = np.mean(NPV_GA)
NPV_std_GA = np.std(NPV_GA)


dist_GA = norm(NPV_mean_GA, NPV_std_GA)

values_GA = [value_GA for value_GA in range(900000, 1000000, 1000)]
probabilities_GA = [dist_GA.pdf(value) for value in values_GA]

NPV_mean_FF = np.mean(NPV_FF)
NPV_std_FF = np.std(NPV_FF)


dist_FF = norm(NPV_mean_FF, NPV_std_FF)

values_FF = [value for value in range(0, 600000,1000)]
probabilities_FF = [dist_GA.pdf(value) for value in values_FF]

plt.hist(NPV_GA, bins = 10, density=True)
plt.plot(values_GA, probabilities_GA)


plt.hist(NPV_FF, bins = 10, density=True)
plt.plot(values_FF, probabilities_FF)
plt.yscale('log')

plt.show()