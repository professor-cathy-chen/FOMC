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
dates = pd.read_csv('D:/paper/dabao/ra/date_full.csv')


date1 = {}
date2 = {}
dds = [i[:-4] for i in os.listdir('D:/paper/dabao/ra/data')]

for i in dates.index:
    day = dates.loc[i, 'FOMC end date']
    d0 = day[:4] + '-' + day[5:7] + '-' + day[8:]

    day = dates.loc[i, 'before 1']
    d1 = day[:4] + '-' + day[5:7] + '-' + day[8:]

    day = dates.loc[i, 'after 1']
    d2 = day[:4] + '-' + day[5:7] + '-' + day[8:]

    if d1 in dds and d2 in dds:
        date1[d1] = pd.read_csv(os.path.join('D:/paper/dabao/ra/data', d1 + '.csv'), index_col=0)
        date2[d2] = pd.read_csv(os.path.join('D:/paper/dabao/ra/data', d2 + '.csv'), index_col=0)

        p = date1[d1]['p(sum1)']
        q = date1[d1]['q(sum1)']

        pp = date2[d2]['p(sum1)']
        qq = date2[d2]['q(sum1)']

        plt.plot(p, label='p')
        plt.plot(q, label='q')
        plt.plot(pp, label='p*')
        plt.plot(qq, label='q*')
        plt.legend()
        plt.savefig('D:/paper/dabao/ra/plot1/'+d0+'.jpg')
        plt.close()

