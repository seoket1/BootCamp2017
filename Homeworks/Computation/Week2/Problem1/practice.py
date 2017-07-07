# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 09:32:47 2017

@author: suket
"""

#  = np.arange(-5,6)**2
y = list(map(lambda x: x ** 2, range(-100,100)))
plt.plot(y) # Draw the line plot
plt.show()

(lambda x,y: x + y)(10, 20)



import numpy as np
from matplotlib import pyplot as plt

plt.ion()
x = np.linspace(1, 4, 100)
plt.plot(x, np.log(x))
plt.plot(x, np.exp(x))
# Clear the figure, then turn interactive mode off.
plt.clf()
plt.ioff()
plt.show()

