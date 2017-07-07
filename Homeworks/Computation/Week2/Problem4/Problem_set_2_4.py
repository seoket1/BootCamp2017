import numpy as np
from pydataset import data as data
from matplotlib import pyplot as plt
import pandas as pd

# Problem 1
diamonds = data("diamonds")
diamonds.head()

cut = diamonds.groupby("cut")
vorder = diamonds.groupby(["cut","color","clarity"])
#vorder.describe()

# Get a group
Fair = cut.get_group("Fair")
Good = cut.get_group("Good")
Very_Good = cut.get_group("Very Good")
Premium = cut.get_group("Premium")
Ideal = cut.get_group("Ideal")

means = cut.mean()
errors = cut.std()
means.loc[:, ["price"]].plot(kind="bar", yerr=errors, title="Mean of Price Data")
plt.xlabel("Daimonds Cut Classification")
plt.ylabel("Price")
plt.show()

'''
By each cut, I wanted to know mean and dispersion of price data.
In this graph, I can know that the mean of priceof Ideal is the lowest, and
the mean of price of Premium is the highest.
'''

# Problem 2


titanic = pd.read_csv("C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem4/titanic.csv")

# 2.1
titanic.groupby("Embarked")
vorder_titanic = titanic.groupby(["Embarked"]).mean()

print(vorder_titanic)

# 2.2
a = titanic.pivot_table("Survived", index =["Embarked","Sex"])

# 2.3 Various ways of interpreting the data
b = titanic.pivot_table("Survived", index =["Embarked","Sex"], columns = "Pclass")
b_count = titanic.pivot_table("Survived", index =["Embarked","Sex"], columns = "Pclass", aggfunc = "count")

age = pd.cut(titanic['Age'], [0,12,18,80])
c = titanic.pivot_table("Survived", index =["Embarked","Sex", age], columns = "Pclass")

# Print
print("\n",a,"\n","\n",b,"\n\n" , b_count, "\n","\n",c)

# Embarked from C has the highest surviving rate. But other embarked city is hard to be comparable.
# the reason is that, from Q, there are not many first and second class passengers.



