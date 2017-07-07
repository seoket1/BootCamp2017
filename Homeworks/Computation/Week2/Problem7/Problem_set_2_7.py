from datetime import datetime
import numpy as np
from pydataset import data as data
from matplotlib import pyplot as plt
import pandas as pd
from datetime import timedelta
import datetime, time 
import statistics
from mpl_toolkits.mplot3d import Axes3D

# (a) 2D Frequency
lipids = pd.read_csv("C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem7/lipids.csv", header = 4)
lipids_diseased = lipids[ lipids["diseased"] == 1]

frequency, section, c = plt.hist(lipids_diseased["chol"], bins = 25)
plt.ylabel("Frequency")
plt.xlabel("Distribution of Cholesterol")
plt.title("2D Histogram of lipids.csv")
plt.show()

print(statistics.median(lipids_diseased["chol"]))

# Median of the data is "212.5". Therefore, median of bins is the NINTH bin.
# the bin with high frequency is the TENTH bin.



# (b) 2D Frequency From https://matplotlib.org/examples/frontpage/plot_3D.html
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = lipids_diseased["chol"], lipids_diseased["trig"]
hist, xedges, yedges = np.histogram2d(x, y, bins=25)

# Construct arrays for the anchor positions of the 16 bars.
# Note: np.meshgrid gives arrays in (ny, nx) so we use 'F' to flatten xpos,
# ypos in column-major order. For numpy >= 1.7, we could instead call meshgrid
# with indexing='ij'.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)

# Construct arrays with the dimensions for the 16 bars.
dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

ax.set_xlabel("chol")
ax.set_ylabel("trig")
ax.set_zlabel("Frequency")

plt.title("3D Histogram of lipids.csv")

plt.show()

'''
Key characteristic is that the frequency of diseased group is higher in both 100~200 chol and more than 400 trig group
compared to others.
'''

# (c) interpretation
'''
If I have only this data, then I could think that high trig group might be riskier than other groups.
However, if we think of Baysian probability, this might not be true.
We have to see more data including non-diseased groups with controlled data.
'''




# 
