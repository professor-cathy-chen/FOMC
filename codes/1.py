import pandas as pd
import os
from datetime import date as dt
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import copy
from multiprocessing import Pool


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
    if len(r) ==0 :
        r = pd.read_csv(os.path.join(path_r, os.listdir(path_r)[0]), index_col=0)
        r = r[r['date'] == r['date'][r['date'] <= date].values[-1]]

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

    # 目的 - 利率插值 函数
    f = interpolate.interp1d(r['days'], r['rate'] / 100, 'linear')

    # 对每一个期权contract进行计算 对应的属性
    for i in option.index:
        exdate = option.loc[i, 'exdate']
        option.loc[i, 'day'] = (dt(int(exdate[:4]), int(exdate[5:7]), int(exdate[8:])) - dt(int(date[:4]), int(date[5:7]), int(date[8:]))).days
        option.loc[i, 'rate'] = f(option.loc[i, 'day'])
        option.loc[i, 'forward_delta'] = option.loc[i, 'delta'] * np.exp(-(option.loc[i, 'rate'] - option.loc[i, 'dividend']) * option.loc[i, 'day'] / 365)
        option.loc[i, 'maturity'] = np.log(option.loc[i, 'day'] / 365)

    # 返回目标的原始矩阵 - res 表示的是 有用的 option contracts for specific date
    res = option[['forward_delta', 'maturity', 'moneyness', 'impl_volatility']]
    return res, f(30), d['rate'].values[0] / 100, stock['close'].values[0]


def extract_q(date):
    raw, r, d, s = extract_option(date)

    number_select = 1000
    mm = (np.array(range(number_select+1)) * 0.48/number_select - 0.24)
    dx = 0.5
    mf = np.log1p(np.expm1(mm) + dx/s)
    ma = np.log1p(np.expm1(mm) - dx/s)
    mmmm = [*mm, *ma, *mf]
    mmmm.sort()

    gp = s * np.exp(mm)[1:] - s * np.exp(mm)[:-1]

    target = pd.DataFrame(np.nan, columns=['moneyness', 'maturity'], index=mmmm)
    target['moneyness'] = mmmm
    target['maturity'] = np.log(1 / 12)

    res = volsurface(raw, target, 2)
    res['call'] = np.nan
    res['q'] = np.nan
    res['strike'] = np.exp(res['moneyness']) * s

    for i in res.index:
        res.loc[i, 'call'] = BS_CALL(s, res.loc[i, 'strike'], 1 / 12, r, d, res.loc[i, 'impl_volatility'])

    for i in res.index[1:-1]:
        if i in mm:
            sy = np.log1p(np.expm1(i) - dx/s)
            xy = np.log1p(np.expm1(i) + dx/s)
            res.loc[i, 'q'] = np.exp(r * 1 / 12) * (res.loc[sy, 'call'] + res.loc[xy, 'call'] - 2 * res.loc[i, 'call'])/dx**2
            if list(mm).index(i) < (len(mm)-1):
                res.loc[i, 'q(sum1)'] = res.loc[i, 'q'] * gp[list(mm).index(i)]/(dx*2)
            else:
                res.loc[i, 'q(sum1)'] = 0
    res = res.loc[~pd.isnull(res['q']), ]
    print(f'{date} - {res.sum(axis=0)["q(sum1)"]}')
    return res


def add_p(data, e):
    data['weight'] = np.exp(e * data['moneyness'])
    sum_value = np.sum(data['q(sum1)'].values * data['weight'].values)
    data['p(sum1)'] = data['weight'].values * data['q(sum1)'].values / sum_value

    plt.plot(data['q(sum1)'], label = 'q')
    plt.plot(data['p(sum1)'], label = 'p')
    plt.legend()
    plt.show()
    plt.close()

    return data


def loop(dd):
    data = extract_q(dd)
    data = add_p(data, 1.5)
    data.to_csv('D:/paper/dabao/ra/data/' + dd + '.csv')


if __name__ == '__main__':

    ss = 'E:/database/option/option price'
    # 所有有option的dates --- days
    days = np.array([i[:-4] for i in os.listdir(ss)])
    # 所有FOMC的dates -- dates
    dates = pd.read_csv('D:/paper/dabao/ra/date_full.csv')
    dates = pd.DataFrame(np.array(dates.stack()), columns=['date'])['date']

    pool = Pool(30)

    # 对每个FOMC的日期进行循环
    for day in dates.values[~pd.isnull(dates.values)]:
        date = day[:4] + '-' + day[5:7] + '-' + day[8:]   # ‘2000-02-02’ --- sample

        if date not in days:
            day_list = [days[days > date][0], days[days < date][-1]]   # 选择上一个 or 下一个有效日期
        else:
            day_list = [date]

        for dd in day_list:
            if dd+'.csv' not in os.listdir('D:/paper/dabao/ra/data'):
                pool.apply_async(loop, args=(dd,))

    pool.close()
    pool.join()

