import numpy as np
from matplotlib import pyplot as plt

# 8.1
x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)

f = 5*x - 4*y

y1 = (1/3) * (2*x + 4)
y2 = (1/6) * (x-1)
y3 = 6 - x

plt.plot(x, y1, label = "y1")
plt.plot(x, y2, label = "y2")
plt.plot(x, y3, label = "y3")
plt.plot(x, np.zeros(x.size), label = "x=0")
plt.plot(np.zeros(y.size), y, label = "y=0")
plt.legend()
plt.text(0,3, 'feasible set')
plt.show()

# 8.2
#(i)
x = np.linspace(0, 10, 1000)
y = np.linspace(0, 10, 1000)

f = 3*x + y

y1 = (1/3) * (15 - x)
y2 = (1/3) * (-2*x + 18)
y3 = x - 4


plt.plot(x, y1, label = "y1")
plt.plot(x, y2, label = "y2")
plt.plot(x, y3, label = "y3")
plt.plot(x, np.zeros(x.size), label = "x=0")
plt.plot(np.zeros(y.size), y, label = "y=0")
plt.legend()
plt.text(0,3, 'feasible set')
plt.show()

#(ii)
x = np.linspace(0, 30, 1000)
y = np.linspace(0, 30, 1000)

f = 4*x + 6*y

y1 = x + 11
y2 = -x + 27
y3 = (1/5)*(-2*x + 90)


plt.plot(x, y1, label = "y1")
plt.plot(x, y2, label = "y2")
plt.plot(x, y3, label = "y3")
plt.plot(x, np.zeros(x.size), label = "x=0")
plt.plot(np.zeros(y.size), y, label = "y=0")
plt.legend()
plt.text(0,3, 'feasible set')
plt.show()

# 8.7
#(ii)
x = np.linspace(0, 5, 1000)
y = np.linspace(0, 5, 1000)

f = 5*x + 2*y

y1 = (1/3)*(15 - 5*x)
y2 = (1/5)*(15-3*x)
y3 = (1/3)*(4*x + 12)


plt.plot(x, y1, label = "y1")
plt.plot(x, y2, label = "y2")
plt.plot(x, y3, label = "y3")
plt.plot(x, np.zeros(x.size), label = "x=0")
plt.plot(np.zeros(y.size), y, label = "y=0")
plt.legend()
plt.text(4,3, 'infeasible')
plt.show()



















