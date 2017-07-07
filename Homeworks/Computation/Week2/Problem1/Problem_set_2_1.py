import numpy as np
from matplotlib import pyplot as plt

# Problem 1
n_array = np.arange(100, 1100, 100)
y_var = np.zeros(10)

for l in range(10):
    n = n_array[l]
    y = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            y[i, j] = np.random.normal()
    
    # Calculating the mean of each row of matrix n*n
    y_mean = np.zeros(n)
    for k in range(n):      
        y_mean[k] = np.mean(y[:,][k])
    
    # Calculating the variance of the means
    y_var[l] = np.var(y_mean)
    
    

plt.axis([100, 1000, -0.01, 0.01]) # [xmin, xmax, ymin, ymax] plt.show()
plt.plot(n_array, y_var)        
plt.show()
                
# Problem 2
x = np.arange(-2 * np.pi, 2 * np.pi, 0.01)
y = np.sin(x)

plt.plot(x, y)
#plt.axis(-2*np.pi, 2*np.pi, -1, 1)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.show()

y = np.cos(x)
plt.plot(x, y)
#plt.axis(-2*np.pi, 2*np.pi, -1, 1)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.show()


y = np.arctan(x)
plt.plot(x, y)
#plt.axis(-2*np.pi, 2*np.pi, -1, 1)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.show()


# Problem 3
import numpy as np
from matplotlib import pyplot as plt

x_2 = np.arange(1, 6, 0.0001)
y_2 = 1/(x_2 - 1)

x_1 = np.arange(-2, 1, 0.0001)
y_1 = 1/(x_1 - 1)

plt.axis([-2, 6, -6, 6])
plt.plot(x_1, y_1, 'm--', linewidth=6)
plt.plot(x_2, y_2, 'm--', linewidth=6)
plt.show()

# Problem 4

x = np.arange(0, 2 * np.pi, 0.01)

plt.suptitle("Various Functions")


y_sinx = np.sin(x)
plt.axis([0, 2 * np.pi, -2, 2])

plt.subplot(221)
plt.plot(x, y_sinx, 'g-', linewidth=6)
plt.title("y_sinx")


y_sin2x = np.sin(2 * x)
plt.axis([0, 2 * np.pi, -2, 2])

plt.subplot(222)
plt.plot(x, y_sin2x, 'r--', linewidth=6)
plt.title("y_sin2x")


y_2sinx = 2 * np.sin(x)
plt.axis([0, 2 * np.pi, -2, 2])

plt.subplot(223)
plt.plot(x, y_2sinx, 'b--', linewidth=6)
plt.title("y_2sinx")

y_2sin2x = 2 * np.sin(2 * x)
plt.axis([0, 2 * np.pi, -2, 2])
plt.subplot(224)
plt.plot(x, y_2sin2x, 'm:', linewidth=6)
plt.title("y_2sin2x")

# For good view
plt.tight_layout()

plt.show()




# Problem 5
data = np.load("C:/Users/suket/Desktop/Homeworks/Computation/Week2/problem1/FARS.npy")

# Draw a scatter plot of x against y, using a circle marker.

longitudes = data[:,1]
latitudes = data[:,2]

plt.figure(figsize = (12, 6))
plt.subplot(121)
plt.axis("equal")
plt.plot(longitudes, latitudes, 'k,')
plt.xlabel("longitudes")
plt.ylabel("latitudes")


# Draw a histogram to display the distribution of the data in x.
plt.subplot(122)
plt.style.use("ggplot")
plt.hist(data[:,0], bins=np.arange(-0.5, 24.5), edgecolor="k") # Or, equivalently,
plt.xlabel("The Hour of the Day")

plt.show()



# Problem 6
x = np.arange(-2 * np.pi, 2 * np.pi, 0.01)
y = x.copy()

X, Y = np.meshgrid(x, y)
Z = (np.sin(X) / X) * (np.sin(Y) / Y)

# Plot the heat map of f over the 2-D domain.
plt.subplot(121)
plt.pcolormesh(X, Y, Z, cmap="magma")
plt.colorbar()
plt.xlim(-np.pi, np.pi)
plt.ylim(-np.pi, np.pi)

# Plot a contour map of f with 10 level curves.
plt.subplot(122)
plt.contour(X, Y, Z, 10, cmap="Spectral")
plt.colorbar()
plt.xlim(-np.pi, np.pi)
plt.ylim(-np.pi, np.pi)

plt.show()



















