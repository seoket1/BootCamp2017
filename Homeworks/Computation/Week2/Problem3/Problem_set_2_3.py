import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import binom
import math
from scipy import stats

# Problem 1
data_1_1 = np.ones(5)
data_1_1 = data_1_1 * (-3)

s1 = pd.Series(data_1_1, index = np.arange(2, 12, 2))

data_1_2 = {'Bill': 31, 'Sarah': 28, 'Jane': 34, 'Joe': 26}
s2 = pd.Series(data_1_2)

print(s1)
print(s2)

# Problem 2
for i in range(5):
    N = 100 # length of random walk
    s = np.zeros(N)
    s[1:] = np.random.binomial(1, 0.5, size=(N-1,))*2-1 #coin flips 1 or -1
    s = pd.Series(s)
    s = s.cumsum() # random walk
    s.plot()
    plt.ylim([-50, 50])
plt.show()

# biased random walks
for i in range(5):
    N = 100 # length of random walk
    s = np.zeros(N)
    s[1:] = np.random.binomial(1, 0.51, size=(N-1,))*2-1 #coin flips 1 or -1
    s = pd.Series(s)
    s = s.cumsum() # random walk
    s.plot()
    plt.ylim([-50, 50])
plt.show()

for i in range(5):
    N = 10000 # length of random walk
    s = np.zeros(N)
    s[1:] = np.random.binomial(1, 0.51, size=(N-1,))*2-1 #coin flips 1 or -1
    s = pd.Series(s)
    s = s.cumsum() # random walk
    s.plot()
    plt.ylim([-500, 500])
plt.show()
    
for i in range(5):
    N = 100000 # length of random walk
    s = np.zeros(N)
    s[1:] = np.random.binomial(1, 0.51, size=(N-1,))*2-1 #coin flips 1 or -1
    s = pd.Series(s)
    s = s.cumsum() # random walk
    s.plot()
    plt.ylim([-4000, 4000])
plt.show()

# Problem 3
#build toy data for SQL operations
name = ['Mylan', 'Regan', 'Justin', 'Jess', 'Jason', 'Remi', 'Matt', 'Alexander', 'JeanMarie']
sex = ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'M', 'F']
age = [20, 21, 18, 22, 19, 20, 20, 19, 20]
rank = ['Sp', 'Se', 'Fr', 'Se', 'Sp', 'J', 'J', 'J', 'Se']
ID = range(9)
aid = ['y', 'n', 'n', 'y', 'n', 'n', 'n', 'y', 'n']
GPA = [3.8, 3.5, 3.0, 3.9, 2.8, 2.9, 3.8, 3.4, 3.7]
mathID = [0, 1, 5, 6, 3]
mathGd = [4.0, 3.0, 3.5, 3.0, 4.0]
major = ['y', 'n', 'y', 'n', 'n']
studentInfo = pd.DataFrame({'ID': ID, 'Name': name, 'Sex': sex, 'Age': age, 'Class': rank})
otherInfo = pd.DataFrame({'ID': ID, 'GPA': GPA, 'Financial_Aid': aid})
mathInfo = pd.DataFrame({'ID': mathID, 'Grade': mathGd, 'Math_Major': major})

'''
# SELECT Name FROM studentInfo WHERE Class = 'J' OR Class = 'Sp'
studentInfo[studentInfo['Class'].isin(['J','Sp'])]['Name']
'''

# SELECT ID, Name from studentInfo WHERE Age > 19 AND Sex = 'M'
Answer3 = studentInfo[ (studentInfo['Age'] > 19) & (studentInfo['Sex'] == 'M')][['ID', 'Name']]

print(Answer3)

# Problem 4
'''
# SELECT * FROM studentInfo INNER JOIN mathInfo ON studentInfo.ID = mathInfo.ID
pd.merge(studentInfo, mathInfo, on='ID') # INNER JOIN is the default
'''

Problem4Info = pd.merge(studentInfo, otherInfo, on='ID', how='outer')[studentInfo['Sex'] == 'M'][['ID','Age','GPA']]
print(Problem4Info)

# Problem 5
crime_data = pd.read_csv("C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem3/crime_data.txt", index_col = "Year")

crime_data['crime_rate'] = crime_data["Total"]/crime_data["Population"]

year = crime_data.index.values
plt.plot(year, crime_data['crime_rate'])
plt.show()

sorted_data = crime_data.sort_values('crime_rate', ascending = False)

# List 5 years with highest crime rate.
for i in range(5):
    print(sorted_data.index[i])

# average of Burglary and total crime.
Burglary_avr = crime_data["Burglary"].mean()
Total_avr = crime_data["Total"].mean()

print(Burglary_avr)
print(Total_avr)

print(crime_data[ (crime_data['Total'] < Total_avr) & (crime_data['Burglary'] \
    > Burglary_avr)])

plt.plot(crime_data['Population'], crime_data['Murder'])
plt.show()

Selected_data = crime_data[(crime_data.index.values >= 1980) & \
        (crime_data.index.values < 1990)][["Population", "Violent","Robbery"]]
Selected_data.to_csv("C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem3/crime_subset.txt")




















