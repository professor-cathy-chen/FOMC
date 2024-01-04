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

dds = [i[:-4] for i in os.listdir('D:/paper/dabao/ra/data')]

entropy = pd.DataFrame(np.nan, columns=dates.columns, index = dates.index)
mj = pd.DataFrame(np.nan, columns=dates.columns, index = dates.index)

for i in entropy.index:
    for j in entropy.columns:
        day = dates.loc[i, j]
        d = day[:4] + '-' + day[5:7] + '-' + day[8:]
        if d in dds:
            file = pd.read_csv(os.path.join('D:/paper/dabao/ra/data', d + '.csv'), index_col=0)
            temp = file['p(sum1)'].values
            hp = - np.nansum(temp * np.log(temp))
            entropy.loc[i, j] = hp
            mj.loc[i, j] = file.sum(axis=0)['q(sum1)']

entropy['d(Hp-FOMC)'] = entropy['after 1'] - entropy['before 1']
entropy['d(Hp-pre FOMC)'] = entropy['before 1'] - entropy['before 2']
entropy.index = dates['FOMC end date'].values

mj['d(Hp-FOMC)'] = mj['after 1'] - mj['before 1']
mj['d(Hp-pre FOMC)'] = mj['before 1'] - mj['before 2']
mj.index = dates['FOMC end date'].values

entropy.to_csv('D:/paper/dabao/ra/entropy.csv')
mj.to_csv('D:/paper/dabao/ra/mj.csv')


entropy.mean(axis=0)
mj.mean(axis=0)


from nltk.corpus import stopwords