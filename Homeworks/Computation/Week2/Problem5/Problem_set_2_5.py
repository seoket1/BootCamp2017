from datetime import datetime
import numpy as np
from pydataset import data as data
from matplotlib import pyplot as plt
import pandas as pd
from datetime import timedelta
import datetime, time

# Problem 1
djia = pd.read_csv("C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem5/DJIA.csv",\
                   index_col = "DATE")

djia.index = pd.to_datetime(djia.index)
djia = djia.replace('.', np.nan)
djia["VALUE"] = djia["VALUE"].astype(float)
print(djia)

# Problem 2
paychecks = pd.read_csv("C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem5/paychecks.csv",\
                        header = -1, names = ["paychecks"])

wom_1fri = pd.date_range(start='3/13/2008', periods = 47, freq="WOM-1FRI")
wom_3fri = pd.date_range(start='3/13/2008', periods = 46, freq="WOM-3FRI")
combined_date_range = wom_1fri.union(wom_3fri)

paychecks = paychecks.set_index(combined_date_range)

print(paychecks)

# Problem 4
finances = pd.read_csv("C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem5/finances.csv",\
                        header = 0)

finances = finances.set_index(pd.period_range("1978-09", periods = 84, freq="Q-AUG"))

print(finances)

# Problem 5
website_traffic = pd.read_csv("C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem5/website_traffic.csv",\
                        header = 0)

website_traffic["ENTER"] = pd.to_datetime(website_traffic["ENTER"])
website_traffic["LEAVE"] = pd.to_datetime(website_traffic["LEAVE"])
website_traffic.index = pd.to_datetime(website_traffic["ENTER"])
website_traffic["Duration"] = website_traffic["LEAVE"] - website_traffic["ENTER"]

#Daily_avr_visit_duration_seonconds
website_traffic["Duration_new"] = 0

website_traffic["Duration_new"] = np.nan
website_duration = []
for i in range(len(website_traffic)):
    temp = website_traffic["Duration"][i].total_seconds()
    website_duration.append(temp)

website_traffic["Duration_new"] = website_duration
    
Daily_avr_visit_duration = website_traffic.resample("D", how = "mean")

print("\nDaily_avr_visit_duration_seconds\n", Daily_avr_visit_duration)

#Totla_number_of_visits_per_hour
website_traffic["Number"] = 1
Totla_number_of_visits_per_hour = website_traffic.resample('H', how = "sum")
print(Totla_number_of_visits_per_hour)


# Problem 6
djia_shift_1 = djia.shift(1)
djia_change_day = djia_shift_1 - djia

VALUE = djia_change_day.groupby("VALUE")
vorder_VALUE = djia_change_day.groupby(["VALUE"])

#Daily max and min
daily_max = max(djia_change_day.groupby("VALUE"))
daily_min = min(djia_change_day.groupby("VALUE"))

print(daily_max)
print(daily_min)

#Monthly max and min
djia_month = djia.resample('M').mean()
djia_month.index = djia_month.index.to_period("M")
djia_change_month = djia_month.shift(1) - djia_month

monthly_max = max(djia_change_month.groupby("VALUE"))
monthly_min = min(djia_change_month.groupby("VALUE"))

print(monthly_max)
print(monthly_min)

# Problem 7
djia_for_graph = djia
djia["WINDOW30"] = djia_for_graph["VALUE"].rolling(window = 30).mean() 
djia["WINDOW365"] = djia_for_graph["VALUE"].rolling(window = 365).mean()
djia_for_graph["SPAN30"] = djia_for_graph["VALUE"].ewm(span = 30).mean()
djia_for_graph["SPAN365"] = djia_for_graph["VALUE"].ewm(span = 365).mean()

djia.plot()
plt.title("Dow Jones Graph")
plt.xlabel("Time")
plt.ylabel("VALUE")
plt.show()



