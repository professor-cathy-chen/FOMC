import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = 'D:/paper/ra/data_11'

for name in os.listdir(path):
    file = pd.read_csv(os.path.join(path, name), index_col=0)
    file['sum_pdf'] = file['q'] * (0.48 / (file.shape[0] - 1))
    file['cum_pdf'] = file['sum_pdf'].rolling(window=file.shape[0], min_periods=1,  axis=0).sum()
    plt.plot(file['q'])
    plt.show()
    print('-')

