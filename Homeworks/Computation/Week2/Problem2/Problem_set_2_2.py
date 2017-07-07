import numpy as np
from matplotlib import pyplot as plt
from scipy.special import binom
import math
from scipy import stats

#Problem 1
data = np.load('C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem2/anscombe.npy')



# I
plt.subplot(221)
plt.plot(data[:,0], data[:,1], 'o', markersize = 10)
plt.xlim(0, 20)
x = np.arange(0, 20, 0.01)
y = (1/2) * x + 3
plt.plot(x, y)
plt.title("maybe real linear case")
plt.show()

# II
plt.subplot(222)
plt.plot(data[:,2], data[:,3], 'o', markersize = 10)
plt.xlim(0, 20)
x = np.arange(0, 20, 0.01)
y = (1/2) * x + 3
plt.plot(x, y)
plt.title("maybe non-linear case.") # Even though regression is same, it is non-linear case.

plt.show()

#III
plt.subplot(223)
plt.plot(data[:,4], data[:,5], 'o', markersize = 10)
plt.xlim(0, 20)
x = np.arange(0, 20, 0.01)
y = (1/2) * x + 3
plt.plot(x, y)
plt.title("One outlier there") # By the one outlier, it does not represent data property.
plt.show()

#IV
plt.subplot(224)
plt.plot(data[:,6], data[:,7], 'o', markersize = 10)
plt.xlim(0, 20)
x = np.arange(0, 20, 0.01)
y = (1/2) * x + 3
plt.plot(x, y)
plt.title("Too many points from one x") # In this case, we cannot trust regression.

plt.show()


# Problem 2
x = np.arange(0, 1, 0.01)

count = 1
for n in range(4):
    for v in range(4):
        plt.subplot(4, 4, count)
        plt.plot(x, binom(n, v) * ((x) ** v) * (1 - x)**(n-v), lw=5)
        plt.axis([0, 1, 0, 1])

        # Turn off extra tick marks and axis labels.
        plt.tick_params(which="both", top="off", right="off")
        if n < 3: # Remove x-axis label on upper plots.
            plt.tick_params(labelbottom="off")
        if v > 0: # Remove y-axis label on right plots.
            plt.tick_params(labelleft="off")
        
        plt.title("n =" + str(n) + "  ,v =" + str(v))
        
        plt.tight_layout()
        
        count += 1



# Problem 3
data_3 = np.load('C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem2/MLB.npy')
height, weight, age = data_3[:, 0], data_3[:, 1], data_3[:, 2]


slope, intercept, r, p, std = stats.linregress(height, weight)

y = intercept + height * slope

plt.subplot(221) # Plot length against width.
plt.scatter(height, weight, s=15)
plt.grid()
plt.ylabel("weight")
plt.tick_params(labelbottom="off")
plt.plot(height, y, "k")

# Continued on the next page.
plt.subplot(222) # Set the marker color to the height.
plt.scatter(height, weight, c=age, s=15)
cbar = plt.colorbar()
cbar.set_label("age")
plt.grid()
plt.tick_params(labelbottom="off", labelleft="off")
plt.plot(height, y, "k")

plt.subplot(223) # Set the marker size to half the volume.
plt.scatter(height, weight, s=15, alpha=0.7)
plt.grid()
plt.xlabel("height")
plt.ylabel("weight")
plt.plot(height, y, "k")

plt.subplot(224) # Use color and marker size together.
plt.scatter(height, weight, c=age, s=15., alpha=.7)
cbar = plt.colorbar()
cbar.set_label("age")
plt.grid()
plt.tick_params(labelleft="off")
plt.xlabel("height")
plt.plot(height, y, "k")


# Problem 4
plt.ioff()
plt.clf()
data_4 = np.load('C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem2/earthquakes.npy')

'''
plt.subplot(131) # Draw a regular histogram.
plt.hist(data, bins=30)
plt.subplot(132) # Draw a clean histogram.
plt.hist(data, bins=30, lw=0, histtype="stepfilled")
plt.tick_params(left="off", top="off", right="off", labelleft="off")
plt.subplot(133) # Convert the histogram to a line plot.
freq, bin_edges = np.histogram(data, bins=30)
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
plt.plot(bin_centers, freq, 'k-', lw=4) 
plt.tick_params(left="off", top="off", right="off", labelleft="off")
'''

year, magnitude, longitude, latitude = np.load("C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem2/earthquakes.npy").T

year_int = year.astype(int)
slope_e, intercept_e, r_e, p_e, std_e = stats.linregress(year, magnitude)

y_magnitude = intercept_e + year * slope_e

plt.scatter(year_int, magnitude)
plt.plot(year, y_magnitude)
plt.xlabel("Year")
plt.ylabel("Magnitude")
plt.title("Scatter Plot")
plt.show()

plt.style.use("ggplot")
plt.hist(year_int, bins = 30, edgecolor="k") 
plt.xlabel("Year")
plt.title("How many earthquakes happened every year?")
plt.show()

plt.style.use("ggplot")
plt.hist(magnitude, bins = 30, edgecolor="k")
plt.title("How often do stronger earthquakes happen compared to weaker ones?")
plt.show()

plt.scatter(longitude, latitude, c = magnitude, s = magnitude ** 3, alpha = 0.7)
cbar = plt.colorbar()
cbar.set_label("magnitude")
plt.grid()
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.show()


# Problem 5
x = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, x.copy())
Z = (1-X)**2 + 100*(Y-X**2)**2

plt.subplot(221) # Plot a heat map of f.
plt.pcolormesh(X, Y, Z, cmap="viridis")
plt.colorbar()

plt.subplot(222) # Plot a contour map with 6 level curves.
plt.contour(X, Y, Z, 15, cmap="viridis")
plt.colorbar()

plt.subplot(223) # Plot a filled contour map with 12 levels.
plt.contourf(X, Y, Z, 15, cmap="viridis")
plt.colorbar()

plt.subplot(224) # Plot specific level curves and a heat map.
plt.contour(X, Y, Z, [0.0001, 0.001, 0.01, 0.1, 1], colors="white")
plt.pcolormesh(X, Y, Z, cmap="viridis")
plt.colorbar()


# Problem 6
pop_2015, gdp_2015, male_height_avr, female_height_avr = np.load('C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem2/countries.npy').T

countries = ["Austria", "Bolivia", "Brazil", "China",
"Finland", "Germany", "Hungary", "India",
"Japan", "North Korea", "Montenegro", "Norway",
"Peru", "South Korea", "Sri Lanka", "Switzerland",
"Turkey", "United Kingdom", "United States", "Vietnam"]


positions = np.arange(len(countries))

plt.figure(figsize = (15,10)) 


plt.subplot(221)
plt.barh(positions, pop_2015, align="center")
plt.yticks(positions, countries)
plt.title("population 2015")

plt.subplot(222)
plt.barh(positions, gdp_2015, align="center")
plt.yticks(positions, countries)
plt.title("gdp 2015")


plt.subplot(223)
plt.barh(positions, male_height_avr, align="center")
plt.yticks(positions, countries)    
plt.xlim(140, 190)
plt.title("male height average 2015")
    
           
plt.subplot(224)
plt.barh(positions, female_height_avr, align="center")
plt.yticks(positions, countries)      
plt.xlim(140, 190)  
plt.title("female height average 2015")

plt.tight_layout()
plt.show()


male_int = male_height_avr.astype(int)
female_int = female_height_avr.astype(int)

plt.hist(male_int)
plt.title("male height average distribution")
plt.xlim(140, 190)
plt.show()

plt.hist(female_int)
plt.title("female height average distribution")
plt.xlim(140, 190)
plt.show()

plt.scatter(gdp_2015, male_height_avr)
plt.title("male height average distribution vs GDP")
plt.show()

plt.scatter(gdp_2015, female_height_avr)
plt.title("female height average distribution vs GDP")
plt.show()

plt.scatter(gdp_2015/pop_2015, male_height_avr)
plt.title("male height average distribution vs GNP")
plt.show()

plt.scatter(gdp_2015/pop_2015, female_height_avr)
plt.title("female height average distribution vs GNP")
plt.show()

'''
Explanation:
    Height is not very much related to GDP. 
    Maybe it is more related to GNP.
'''
    
    
    









