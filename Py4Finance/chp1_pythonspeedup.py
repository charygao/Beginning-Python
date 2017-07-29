#!/usr/bin/env python
# -*-encoding:utf-8-*-

import time
 

#demo1:耗时统计
start = time.clock()

loops = 25000000
from math import *
a = range(1,loops)
def f(x):
 return 3*log(x)+cos(x)**2
r = [f(x) for x in a]

elapsed = (time.clock() - start)
print("Demo1 Time used:",elapsed)



#demo2:耗时统计
import numpy as np 
start = time.clock()

a= np.arange(1,loops)
r= 3*np.log(a)+np.cos(a)**2

elapsed = (time.clock() - start)
print("Demo2 Time used:",elapsed)




#demo3:耗时统计
import numexpr as ne 
start = time.clock()

ne.set_num_threads(1)
f='3*log(a)+cos(a)**2'
r = ne.evaluate(f)

elapsed = (time.clock() - start)
print("Demo3 Time used:",elapsed)




#demo4:耗时统计
start = time.clock()

ne.set_num_threads(4)
r = ne.evaluate(f)

elapsed = (time.clock() - start)
print("Demo4 Time used:",elapsed)