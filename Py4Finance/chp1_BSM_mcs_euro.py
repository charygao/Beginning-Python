#!/usr/bin/env python
# -*-encoding:utf-8-*-

#欧式看涨期权的蒙特卡洛估值
#Monte carlo Valuation of European call option
#in Black-Scholes-Merton model
#bsm_mcs_euro.py


import numpy as np 

#Parameters 
s0 = 100.  #initial index level
K=105.   #strike price
T=1.0   # time-to-maturity
r=0.05  #riskless short rate
sigma = 0.2  #volatility

#number of  simulations
I = 100000 

#Valuation algorithm
z= np.random.standard_normal(I) #pseudorandom numbers
ST = s0*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*z) #index value at maturity
hT = np.maximum(ST-K,0) #inner values at maturity
C0 = np.exp(-r*T)*np.sum(hT)/I

#output
print 'Value of the European Call Option: %5.3f'%C0
