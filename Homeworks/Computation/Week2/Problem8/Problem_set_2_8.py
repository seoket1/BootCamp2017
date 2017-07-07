from datetime import datetime
import numpy as np
from pydataset import data as data
from matplotlib import pyplot as plt
import pandas as pd
from datetime import timedelta
import datetime, time 
import statistics
from mpl_toolkits.mplot3d import Axes3D

# from 1/1/1939
payems = pd.read_csv("C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem8/payems.csv", header = 15, names=["date", "payems"])

# For managing First segment data.. It is different from others.
payems_0 = pd.read_csv("C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem8/payems.csv", header = 5, names=["date", "payems"])
first_period_index = pd.date_range(start = "7/1/1929", end = "7/1/1936", freq="12MS")

payems_0 = payems_0[0:8]
payems_0.index = first_period_index

# Setting time index
payems.index = pd.date_range(start = "1/1/1939", end = "10/1/2016", freq="MS")

# Setting segments
payems_list = [payems_0, payems["1936-05":"1944-05"], payems["1944-02":"1952-02"], \
                payems["1947-11":"1955-11"], payems["1952-6":"1960-6"],\
                payems["1956-08":"1964-08"], payems["1959-04":"1967-04"], \
                payems["1968-12":"1976-12"], payems["1972-11":"1980-11"], \
                payems["1979-01":"1987-01"], payems["1980-07":"1988-07"], \
                payems["1989-07":"1997-07"], payems["2000-03":"2008-03"], payems["2006-12":"2014-12"]]

# (b) Normalize
peak_date= ["7/1/29", "1/1/39", "2/1/45", "11/1/48", "6/1/53", "8/1/57","4/1/60", "12/1/69","11/1/73",\
            "1/1/80","7/1/81","7/1/90","3/1/01","12/1/07"]


for i in range(len(payems_list)):
    payems_list[i]["Normalized"] = payems_list[i]["payems"] / ( payems_list[i][payems_list[i]["date"] == peak_date[i]]["payems"][0] )
    

# (c), (d), (e), (f), (i) Plot each segments

# For merging data in order to plot on same axis.
Count_index = []
for k in range(97):
    Count_index.append(k)
    
Count_index_for_first_period = []
for m in range(8):
    Count_index_for_first_period.append(11 + 12*m)   

Count_index_for_second_period = []
for n in range(65):
    Count_index_for_second_period.append(n+32)   



for l in range(len(payems_list)):
    if l == 0:
        payems_list[l]["Count"] = Count_index_for_first_period
    elif l == 1:
        payems_list[l]["Count"] = Count_index_for_second_period
    else:
        payems_list[l]["Count"] = Count_index

linestyles = ['-', '--', '-.', ':','-', '--', '-.', ':','-', '--', '-.', ':', '-', '--']

for j in range(len(payems_list)):
    if j == 0:
        plt.plot(payems_list[j]["Count"], payems_list[j]["Normalized"], color = "black", linewidth =5, linestyle = "-", label = peak_date[j] + " Great Depression" )
    elif j == 13:
        plt.plot(payems_list[j]["Count"], payems_list[j]["Normalized"], color = "red", linewidth =5, linestyle = "-", label = peak_date[j] + " Great Recession" )
    else:
        plt.plot(payems_list[j]["Count"], payems_list[j]["Normalized"], linestyle = linestyles[j], label = peak_date[j] )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    
plt.title("Job Percentage of each Segment during Recession Time")
plt.ylabel("Jobs/peak")
plt.xlabel("Time from peak")

# (g) Labeling on x-axis
plt.xticks(np.arange(0, 100, 12), ("-1yr", "peak", "+1yr", "+2yr", "+3yr", "+4yr", "+5yr", "+6yr", "+7yr"))

# (h) dashed line
plt.axvline(x = 12, c='grey', linestyle = "--")
plt.axhline(y = 1, c='grey', linestyle = "--")

# (i)
# I added line setting into plot function above.

plt.show()

# (j) There was "NOT" any worse recessions than Great Recession except Great Depression.

# (k) In terms of nonagricultural jobs, Great Depression was the worst definitely when I see this graph.
# However, we have to see compare with population. This is just a jobs total. 
# If popluation change direction is different, maybe the result can be changed.

























