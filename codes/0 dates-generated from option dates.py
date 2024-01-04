import pandas as pd
import os
from datetime import date as dt
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import copy

ss = 'E:/database/option/option price'
# 所有有option的dates --- days
days = np.array([i[:-4] for i in os.listdir(ss)])
# 所有FOMC的dates -- dates
dates = pd.read_csv('D:/paper/dabao/ra/date before1and2 after1and2.csv')
fulldates = np.array([i[:-4] for i in os.listdir('E:/database/option/option price')])
fulldates = np.array(sorted(fulldates))

final_dates = pd.DataFrame(np.nan, columns=dates.columns, index=dates.index)
final_dates['FOMC end date'] = [day[:4] + '-' + day[5:7] + '-' + day[8:] for day in dates['FOMC end date']]

for i in dates.index:
    try:
        b1 = fulldates[fulldates < final_dates.loc[i, 'FOMC end date']][-1]
        final_dates.loc[i, 'before 1'] = b1
    except:
        pass

    try:
        b2 = fulldates[fulldates < final_dates.loc[i, 'FOMC end date']][-2]
        final_dates.loc[i, 'before 2'] = b2
    except:
        pass

    try:
        a1 = fulldates[fulldates >= final_dates.loc[i, 'FOMC end date']][0]
        final_dates.loc[i, 'after 1'] = a1
    except:
        pass

    try:
        a2 = fulldates[fulldates >=final_dates.loc[i, 'FOMC end date']][1]
        final_dates.loc[i, 'after 2'] = a2
    except:
        pass

for i in final_dates.index:
    for j in final_dates.columns:
        day = final_dates.loc[i,j]
        day = day[:4] + '/' + day[5:7] + '/' + day[8:]
        final_dates.loc[i,j] = day


final_dates.to_csv('D:/paper/dabao/ra/date_full.csv', index=False)
