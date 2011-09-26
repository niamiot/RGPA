        # -*- coding:Utf-8 -*-
import numpy as np
import scipy as sp 
import time 
import pdb
import os
import sys

import matplotlib.pyplot as plt

a=[]
a.append(np.load('RSS.npy'))
a.append(np.load('TOA.npy'))
a.append(np.load('TDOA.npy'))
a.append(np.load('TOARSS.npy'))
a.append(np.load('TDOARSS.npy'))
a.append(np.load('TDOATOA.npy'))
a.append(np.load('TDOATOARSS.npy'))


X=np.array(a)


N =7 


ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, X[:,1], width, color='r')
rects2 = ax.bar(ind+width, X[:,0], width, color='y')

# add some
ax.set_ylabel('mean time (s)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('RSS', 'TOA', 'TDOA', 'TOA + RSS', 'TDOA + RSS', 'TOA + TDOA', 'RSS + TOA + TDOA') )

ax.legend( (rects1[0], rects2[0]), ('RGPA', 'ML-NMS') ,loc ='best')

#def autolabel(rects):
#    # attach some text labels
#    for rect in rects:
#        height = rect.get_height()
#        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
#                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

plt.show()

