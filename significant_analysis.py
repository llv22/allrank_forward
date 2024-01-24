import numpy as np
from scipy import stats

from astropy.table import Table
tab = Table.read('statistics.tex')

# print(tab)
np_array_structured = np.array(tab.as_array())
# print(np_array_structured)

# Example data
baseline = np.array(list(np_array_structured[0])[1:]) * 167
for i in range(1, len(np_array_structured)):
    for j in range(1, len(np_array_structured[i])):
        new_result = np.array([np_array_structured[i][j]]) * 167
        # Perform a two-sample t-test
        t_stat, p_value = stats.ttest_ind(baseline[j-1], new_result)

        # print("T-statistic:", t_stat)
        # print("p-value:", p_value)

        # Interpret the result
        alpha = 0.05
        if p_value < alpha:
            print("Result is statistically significant")
        else:
            print("No significant difference found")
