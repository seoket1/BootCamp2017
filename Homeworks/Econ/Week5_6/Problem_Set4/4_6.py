import pandas_datareader.data as web
import datetime
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import fredpy as fp
import scipy.stats as sts

## For finding variables' name, Clint Hamilton helped.
## For Moving Average codes, refered from https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy ##
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

start = datetime.datetime(1960, 1, 1)
end = datetime.datetime(2011, 1, 1)

gdp = web.DataReader("GDPC1", "fred", start, end)
gdp = np.array(gdp)

GDP = web.DataReader("GDPC1", "fred", start, end)
Cons = web.DataReader("PCEC", "fred", start, end)
Inv = web.DataReader("GPDI", "fred", start, end)
GovS = web.DataReader("W068RCQ027SBEA", "fred", start, end)
NX = web.DataReader("NETEXP", "fred", start, end)
Exp = web.DataReader("EXPGSC1", "fred", start, end)
Imp = web.DataReader("IMPGSC1", "fred", start, end)
Emp = web.DataReader("PAYEMS", "fred", start, end)
Emp = Emp.resample('3M').mean()
Hrs = web.DataReader("PRS88003033", "fred", start, end)
TotalLabor = web.DataReader(
    "MPU4900052", "fred", datetime.datetime(1990, 1, 1), end)
Nwage = web.DataReader("A4102C1Q027SBEA", "fred", start, end)
Rwage = web.DataReader("LES1252881600Q", "fred",
                       datetime.datetime(1980, 1, 1), end)
NominalIR = web.DataReader("B069RC1Q027SBEA", "fred", start, end)
RealIR = web.DataReader("FEDFUNDS", "fred", start, end)
RealIR = RealIR.resample('3M').mean()
MS = web.DataReader("MABMM301USQ189S", "fred", start, end)
RMS = web.DataReader("M1REAL", "fred", start, end)
RMS = RMS.resample('3M').mean()
GDPdef = web.DataReader("GDPDEF", "fred", start, end)
CPI = web.DataReader("CPIAUCSL", "fred", start, end)
CPI = CPI.resample('3M').mean()
SolowRes = web.DataReader("A959RX1A020NBEA", "fred", start, end)
LaborProd = web.DataReader("MPU4900062", "fred", start, end)
Unemp = web.DataReader("UNRATE", "fred", start, end)
Unemp = Unemp.resample('3M').mean()
ParR = web.DataReader("CIVPART", "fred", start, end)
ParR = ParR.resample('3M').mean()
BudgDef = web.DataReader("FYFSD", "fred", start, end)

series = [GDP, Cons, Inv, GovS, NX, Exp, Imp, Emp, Hrs, TotalLabor, Nwage, Rwage,
          NominalIR, RealIR, MS, RMS, GDPdef, CPI, SolowRes, LaborProd, Unemp, ParR, BudgDef]

for i in range(len(series)):
    series[i] = np.array(series[i])

seriesn = ["GDP", "Cons", "Inv", "Gov. S", "NX", "Exp.", "Imp.", "Emp.", "Hrs", "Total Labor", "N wage", "R wage",
           "Nominal IR", "Real IR", "MS", "RMS", "GDPdef", "CPI", "Solow Res", "Labor Prod", "Unemp.", "Par. R", "Budg Def"]

for i, x in enumerate(series):
    print("\n\n")
    print("for ", seriesn[i])
    print("std = ", np.std(x) / np.mean(x))
    print("rel to gdp = ", np.std(x) / np.mean(gdp))

    print("AR = ", sts.pearsonr(series[i][1:], series[i][:-1])[0][0])
    print("corr w/ gdp = ",
          sts.pearsonr(series[i], gdp[-len(series[i]):])[0][0])
    print("\n****Linear Trend Filter****\n")
    MR = moving_average(series[i], n=3)
    plt.plot(range(MR.size), MR, label = seriesn[i])
    plt.legend()
    plt.show()    
    
    print("\n****First Difference****\n")
    FIR = series[i][1:] - series[i][:-1]
    plt.plot(range(FIR.size), FIR, label = seriesn[i])
    plt.legend()
    plt.show()
    
    print("\n****HP filter****\n")
    cy, tr = sm.tsa.filters.hpfilter(series[i], 1600)
    plt.plot(range(tr.size), tr, label = seriesn[i])
    plt.legend()
    plt.show()
    
    print("\n****BP filter****\n")
    cycles = sm.tsa.filters.bkfilter(series[i], 6, 32, 8)
    plt.plot(range(cycles.size), cycles, label = seriesn[i])
    plt.legend()
    plt.show()



