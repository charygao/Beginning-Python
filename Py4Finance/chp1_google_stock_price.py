#!/usr/bin/env python
# -*-encoding:utf-8-*-

import numpy as np 
import pandas as pd 
from pandas_datareader import data, wb
import matplotlib.pyplot as plt

goog = data.DataReader('GOOG',data_source='yahoo',start='3/14/2008',end='7/14/2017')
print goog.tail()


# 实现对比波动率的分析

goog['log_Ret'] = np.log(goog['Close']/goog['Close'].shift(1))
goog['Volatility'] = pd.rolling_std(goog['log_Ret'],window=252)*np.sqrt(252)
print goog.tail()


#绘图
goog[["Close","Volatility"]].plot(subplots=True,color='blue',figsize=(8,6))
plt.show()