import numpy as np
from scipy import stats

from astropy.table import Table
tab = Table.read('statistics.tex')

# print(tab)
np_array_structured = np.array(tab.as_array())
# print(np_array_structured)

# Example data
baseline = np.array(list(np_array_structured[0])[1:])[0:2]
for i in range(1, len(np_array_structured)):
    new_result = np.array(list(np_array_structured[i])[1:])[0:2]
    # Perform a two-sample t-test
    t_stat, p_value = stats.ttest_ind(baseline, new_result)

    # print("T-statistic:", t_stat)
    # print("p-value:", p_value)

    # Interpret the result
    alpha = 0.05
    if p_value < alpha:
        print("Result is statistically significant")
    # else:
    #     print("No significant difference found")
