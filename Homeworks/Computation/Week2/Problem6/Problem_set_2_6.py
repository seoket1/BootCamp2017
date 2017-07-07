from datetime import datetime
import numpy as np
from pydataset import data as data
from matplotlib import pyplot as plt
import pandas as pd
from datetime import timedelta
import datetime, time 


Indi = pd.read_csv("C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem6/Indi.csv", header = 0)
Pitt = pd.read_csv("C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem6/Pitt.csv", header = 0)
Mian = pd.read_csv("C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem6/Mian.csv", header = 0)
Wash = pd.read_csv("C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem6/Wash.csv", header = 0)
Chic = pd.read_csv("C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem6/Chic.csv", header = 0)

City_list = [Indi, Pitt, Mian, Wash, Chic]
leap_year = 1976, 1980, 


for i in range(len(City_list)):
    City_list[i]["DAY"] = np.nan
    day_temp = []    
    for j in range(len(City_list[i])):
        date = datetime.datetime.strptime(str(City_list[i]["DATE"][j]),"%Y%m%d")
        day = date.timetuple().tm_yday
        year = date.timetuple().tm_year
        
        if ((year % 4) ==0 ) & ((year % 100) != 0) | ((year % 400) == 0):
            day = day - 264
        else:
            day = (day - 263)
    
        if ((year % 4) ==0 ) & ((year % 100) != 0) | ((year % 400) == 0):
            if day < 1:
                day += 366
        else:
            if day < 1:
                day += 365
            
        day_temp.append(day)
    
    City_list[i]["DAY"] = day_temp
    City_list[i].index = City_list[i]["DAY"]



''' Just for reference. I do not use these dataframe.
Pitt["DATE"] = pd.to_datetime(pd.Series(Pitt["DATE"]), format="%Y%m%d")
Mian["DATE"] = pd.to_datetime(pd.Series(Mian["DATE"]), format="%Y%m%d")
Wash["DATE"] = pd.to_datetime(pd.Series(Wash["DATE"]), format="%Y%m%d")
Chic["DATE"] = pd.to_datetime(pd.Series(Chic["DATE"]), format="%Y%m%d")
'''

for i in range(len(City_list)):
    if i == 4:  # The Case of Chicago 
        plt.plot(City_list[i].index, City_list[i]["TMAX"], 'o', color = "maroon", markersize = 1)
        plt.plot(City_list[i].index, City_list[i]["TMIN"], 'o', color = "maroon", markersize = 1)
    else:
        plt.plot(City_list[i].index, City_list[i]["TMAX"], 'o', color = "black", markersize = 1)
        plt.plot(City_list[i].index, City_list[i]["TMIN"], 'o', color = "gray", markersize = 1)

plt.ylabel("Temperature(Fahrenheit)")
plt.xlabel("366 days")


born = City_list[0][City_list[0]["DATE"] == 19750122]
little = City_list[1][City_list[1]["DATE"] == 19880714]

plt.plot(born.index, born["TMAX"], marker = '+', color = "yellow", mew = 3, markersize = 3 )
plt.plot(born.index, born["TMIN"], marker = '+', color = "yellow", mew = 3, markersize = 3 )

plt.plot(little.index, little["TMAX"], marker = 'o', color = "yellow", mew = 3, markersize = 3 )
plt.plot(little.index, little["TMIN"], marker = 'o', color = "yellow", mew = 3, markersize = 3 )

plt.annotate('born (TMAX)', xy=(124, 40), xytext=(200, 100), arrowprops=dict(arrowstyle="->"))
plt.annotate('born (TMIN)', xy=(124, 21), xytext=(200, 0), arrowprops=dict(arrowstyle="->"))

plt.annotate("little league all-star team wins regional championship (TMAX)", xy=(298, 92), xytext=(350, 100), arrowprops=dict(arrowstyle="->"))
plt.annotate("little league all-star team wins regional championship (TMAX)", xy=(298, 71), xytext=(300, 0), arrowprops=dict(arrowstyle="->"))


plt.show()



