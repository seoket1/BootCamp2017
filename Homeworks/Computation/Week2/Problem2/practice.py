# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 21:54:52 2017

@author: suket
"""

plt.plot(x, T(n)(x), lw=2)
... plt.axis([0, 1, 0, 1])
...
... # Turn off extra tick marks and axis labels.
... plt.tick_params(which="both", top="off", right="off")
... if n < 6: # Remove x-axis label on upper plots.
... plt.tick_params(labelbottom="off")
... if n % 3: # Remove y-axis label on right plots.
... plt.tick_params(labelleft="off")
... plt.title("n = "+str(n))