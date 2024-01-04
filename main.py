import pandas as pd
import os
from datetime import date as dt
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def volsurface(raw, target, bds):
    '''
    raw:     dataframe - initial data  for call/put with BMS forward delta, log-maturity, log-moneyness
    target:  dataframe - target volatility, with log-maturity, log-moneyness
    bds:     default bandwiths for hx, ht
    '''
    nobs = raw.shape[0]
    sigmax = np.std(raw['moneyness'], ddof=1)
    sigmat = np.std(raw['maturity'], ddof=1)
    hx = bds * (4 / 3) ** (1 / 5) * sigmax / nobs ** (1 / 5)
    ht = bds * (4 / 3) ** (1 / 5) * sigmat / nobs ** (1 / 5)

    new_target = copy.deepcopy(target)
    new_target['impl_volatility'] = np.nan

    for i in new_target.index:
        target_x = target.loc[i, 'moneyness']
        target_t = target.loc[i, 'maturity']
        w = (1 - np.abs(raw['forward_delta'])) * (np.abs(raw['forward_delta']) < 0.8) \
            * np.exp(-np.square(raw['moneyness'] - target_x) / 2 / np.square(hx)) \
            * np.exp(-np.square(raw['maturity'] - target_t) / 2 / np.square(ht))
        w = w / np.sum(w)
        new_target.loc[i, 'impl_volatility'] = np.sum(w * raw['impl_volatility'])

    return new_target


def BS_CALL(S, K, T, r, d, sigma):
    N = norm.cdf
    d1 = (np.log(S/K) + (r - d + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-d*T) * N(d1) - K * np.exp(-r*T) * N(d2)


def extract_option(date):
    path_option = 'E:/database/option/option price'
    path_s = 'E:/database/option/security'
    path_d = 'E:/database/option/index dividend'
    path_r = 'E:/database/option/zero coupon yield curve'
    # option 数据
    option = pd.read_csv(os.path.join(path_option, date + '.csv'), index_col=0)
    # stock 数据
    stock = pd.read_csv(os.path.join(path_s, date + '.csv'), index_col=0)
    option['secid'] = option['secid'].astype(int)
    stock['secid'] = stock['secid'].astype(int)
    # 分红数据
    d = pd.read_csv(os.path.join(path_d, os.listdir(path_d)[0]), index_col=0)
    d['secid'] = d['secid'].astype(int)
    d = d[d['secid'] == 108105]
    d = d[d['date'] == date]
    # 利率数据
    r = pd.read_csv(os.path.join(path_r, os.listdir(path_r)[0]), index_col=0)
    r = r[r['date'] == date]
    if 1 not in r['days']:
        r.loc[max(r.index) + 1, :] = [r.iloc[0, 0], 1.0, r.iloc[0, 2]]
    r.loc[max(r.index) + 2, :] = [r.iloc[-1, 0], 10000.0, r.iloc[-1, 2]]

    # 1 select only S&P 500 index option
    option = option[option['secid'] == 108105]
    stock = stock[stock['secid'] == 108105]
    option['dividend'] = d['rate'].values[0] / 100
    option['stock'] = stock['close'].values[0]
    option['moneyness'] = np.log(option['strike_price'] / 1000 / option['stock'])
    # 2 delete options with non-valid iv
    option = option[~pd.isnull(option['impl_volatility'])]
    # 3 calculate maturity-day & log(maturity in year)
    option['day'] = np.nan
    option['maturity'] = np.nan
    option['rate'] = np.nan

    # Purpose - Interest Rate Interpolation Function
    f = interpolate.interp1d(r['days'], r['rate'] / 100, 'linear')

    # Calculated for each option contract Corresponding property
    for i in option.index:
        exdate = option.loc[i, 'exdate']
        option.loc[i, 'day'] = (dt(int(exdate[:4]), int(exdate[5:7]), int(exdate[8:])) - dt(int(date[:4]), int(date[5:7]), int(date[8:]))).days
        option.loc[i, 'rate'] = f(option.loc[i, 'day'])
        option.loc[i, 'forward_delta'] = option.loc[i, 'delta'] * np.exp(-(option.loc[i, 'rate'] - option.loc[i, 'dividend']) * option.loc[i, 'day'] / 365)
        option.loc[i, 'maturity'] = np.log(option.loc[i, 'day'] / 365)

    # Return the original matrix of the target - res indicates a valid option contracts for specific date
    res = option[['forward_delta', 'maturity', 'moneyness', 'impl_volatility']]
    return res, f(30), d['rate'].values[0] / 100, stock['close'].values[0]


def extract_p(date):
    raw, r, d, s = extract_option(date)

    mm = (np.array(range(601)) / 1000 - 0.3)
    target = pd.DataFrame(np.nan, columns=['moneyness', 'maturity'], index=mm)
    target['moneyness'] = mm
    target['maturity'] = np.log(1 / 12)

    res = volsurface(raw, target, 2)
    res['call'] = np.nan
    res['p'] = np.nan
    res['strike'] = np.exp(res['moneyness']) * s

    for i in res.index:
        res.loc[i, 'call'] = BS_CALL(s, res.loc[i, 'strike'], 1 / 12, r, d, res.loc[i, 'impl_volatility'])

    for i in res.index[1:-1]:
        sy = res.index[res.index.to_list().index(i) - 1]
        xy = res.index[res.index.to_list().index(i) + 1]
        res.loc[i, 'p'] = np.exp(r * 1 / 12) * (
                (res.loc[sy, 'call'] - res.loc[i, 'call']) / (res.loc[i, 'strike'] - res.loc[sy, 'strike']) -
                (res.loc[i, 'call'] - res.loc[xy, 'call']) / (res.loc[xy, 'strike'] - res.loc[i, 'strike'])) / ((res.loc[xy, 'strike'] - res.loc[sy, 'strike']) / 2)

    return res


ss = 'E:/database/option/option price'
# All option dates --- days
days = np.array([i[:-4] for i in os.listdir(ss)])
# All FOMC dates -- dates
dates = pd.read_csv('D:/paper/dabao/date.csv')['date']

# Loop for each FOMC date
for day in dates.values[~pd.isnull(dates.values)]:
    date = day[:4] + '-' + day[5:7] + '-' + day[8:]   # ‘2000-02-02’ --- sample

    if date not in days:
        day_list = [days[days>date][0], days[days<date][-1]]   # Select the previous or next valid date
    else:
        day_list = [date]

    for dd in day_list:
        if dd+'.csv' not in os.listdir('D:/paper/dabao/data'):
            data = extract_p(dd)
            data.to_csv('D:/paper/dabao/data/'+dd+'.csv')


