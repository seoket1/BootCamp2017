import pandas_datareader.data as web
import datetime
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

'''Ex 3'''
# set start and end dates
start = datetime.datetime(1960, 1, 1)
end = datetime.datetime(2017, 1, 1)

# download data from FRED using pandas datareader
# this series is seasonally unadjusted industrial production at monthly frequencies
GDP = web.DataReader("GDPC1", "fred", start, end) 
Consumption = web.DataReader("PCE", "fred", start, end) 
Investment = web.DataReader("GPDI", "fred", start, end)
CPI = web.DataReader("CPALTT01USQ657N", "fred", start, end)

# convert to quarterly frequencies by averaging over the three months in the quarter
GDP = GDP.resample('3M').mean()
Consumption = Consumption.resample('3M').mean()
Investment = Investment.resample('3M').mean()
CPI = CPI.resample('3M').mean()

# take the natural log of the series and convert to numpy array
logGDP = np.log(GDP.values)
logConsumption = np.log(Consumption.values)
logInvestment = np.log(Investment.values)
logCPI = CPI.values  # Exception

GDPfreq, GDPper = sig.periodogram(logGDP, axis=0)
Consumptionfreq, Consumptionper = sig.periodogram(logConsumption, axis=0)
Investmentfreq, Investmentper = sig.periodogram(logInvestment, axis=0)
CPIfreq, CPIper = sig.periodogram(logCPI, axis=0)

plt.plot(GDPfreq[1:], np.log(GDPper[1:]))
plt.title('GDP')
plt.xlabel('frequency')
plt.show()

plt.plot(GDP)
plt.title('GDP')
plt.xlabel('time')
plt.show()

plt.plot(Consumptionfreq[1:], np.log(Consumptionper[1:]))
plt.title('Consumption')
plt.xlabel('frequency')
plt.show()

plt.plot(Consumption)
plt.title('Consumption')
plt.xlabel('time')
plt.show()


plt.plot(Investmentfreq[1:], np.log(Investmentper[1:]))
plt.title('Investment')
plt.xlabel('frequency')
plt.show()

plt.plot(Investment)
plt.title('Investment')
plt.xlabel('time')
plt.show()


plt.plot(CPIfreq[1:], np.log(CPIper[1:]))
plt.title('CPI')
plt.xlabel('frequency')
plt.show()

plt.plot(CPI)
plt.title('CPI')
plt.xlabel('time')
plt.show()

'''Ex 4'''
## GDP ##
GDPcy, GDPtr = sm.tsa.filters.hpfilter(logGDP, 1600)
GDPfreq, GDPcyper = sig.periodogram(GDPcy, axis=0)

plt.plot(GDPfreq[1:], np.log(GDPcyper[1:]))
plt.title('GDP - cyclical')
plt.xlabel('frequency')
plt.show()

nobs = GDPcyper.size
gain = np.zeros(nobs)
for i in range(0, nobs):
    gain[i] = GDPcyper[i] / GDPper[i,0]

plt.plot(GDPfreq[1:], np.log(gain[1:]))
plt.title('Filter Gain')
plt.xlabel('frequency')
plt.show()


plt.plot(range(GDPtr.size), logGDP, 'k-',
         range(GDPtr.size), GDPtr, 'r-')
plt.title('GDP')
plt.xlabel('time')
plt.show()

plt.plot(GDPcy)
plt.title('GDP - cyclical')
plt.xlabel('time')
plt.show()

## Consumption ##
Consumptioncy, Consumptiontr = sm.tsa.filters.hpfilter(logConsumption, 1600)
Consumptionfreq, Consumptioncyper = sig.periodogram(Consumptioncy, axis=0)

plt.plot(Consumptionfreq[1:], np.log(Consumptioncyper[1:]))
plt.title('Consumption - cyclical')
plt.xlabel('frequency')
plt.show()

nobs = Consumptioncyper.size
gain = np.zeros(nobs)
for i in range(0, nobs):
    gain[i] = Consumptioncyper[i] / Consumptionper[i,0]

plt.plot(Consumptionfreq[1:], np.log(gain[1:]))
plt.title('Filter Gain')
plt.xlabel('frequency')
plt.show()


plt.plot(range(Consumptiontr.size), logConsumption, 'k-',
         range(Consumptiontr.size), Consumptiontr, 'r-')
plt.title('Consumption')
plt.xlabel('time')
plt.show()

plt.plot(Consumptioncy)
plt.title('Consumption - cyclical')
plt.xlabel('time')
plt.show()

## Investment ##
Investmentcy, Investmenttr = sm.tsa.filters.hpfilter(logInvestment, 1600)
Investmentfreq, Investmentcyper = sig.periodogram(Investmentcy, axis=0)

plt.plot(Investmentfreq[1:], np.log(Investmentcyper[1:]))
plt.title('Investment - cyclical')
plt.xlabel('frequency')
plt.show()

nobs = Investmentcyper.size
gain = np.zeros(nobs)
for i in range(0, nobs):
    gain[i] = Investmentcyper[i] / Investmentper[i,0]

plt.plot(Investmentfreq[1:], np.log(gain[1:]))
plt.title('Filter Gain')
plt.xlabel('frequency')
plt.show()


plt.plot(range(Investmenttr.size), logInvestment, 'k-',
         range(Investmenttr.size), Investmenttr, 'r-')
plt.title('Investment')
plt.xlabel('time')
plt.show()

plt.plot(Investmentcy)
plt.title('Investment - cyclical')
plt.xlabel('time')
plt.show()

## CPI ##
CPIcy, CPItr = sm.tsa.filters.hpfilter(logCPI, 1600)
CPIfreq, CPIcyper = sig.periodogram(CPIcy, axis=0)

plt.plot(CPIfreq[1:], np.log(CPIcyper[1:]))
plt.title('CPI - cyclical')
plt.xlabel('frequency')
plt.show()

nobs = CPIcyper.size
gain = np.zeros(nobs)
for i in range(0, nobs):
    gain[i] = CPIcyper[i] / CPIper[i,0]

plt.plot(CPIfreq[1:], np.log(gain[1:]))
plt.title('Filter Gain')
plt.xlabel('frequency')
plt.show()


plt.plot(range(CPItr.size), logCPI, 'k-',
         range(CPItr.size), CPItr, 'r-')
plt.title('CPI')
plt.xlabel('time')
plt.show()

plt.plot(CPIcy)
plt.title('CPI - cyclical')
plt.xlabel('time')
plt.show()


'''Ex5'''
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]

Lambda = np.array([100, 400, 1600, 6400, 25600])

## GDP ##
for Lam in Lambda:
    cy, tr = sm.tsa.filters.hpfilter(logGDP, Lam)
    freq, cyper = sig.periodogram(GDPcy, axis=0)
    
    nobs = cyper.size
    gain = np.zeros(nobs)
    for i in range(0, nobs):
        gain[i] = cyper[i] / GDPper[i,0]
        
    plt.plot(range(tr.size), tr, label = Lam)
    print("\n\n *****", Lam, "\n\n") 
    print("mean :", tr.mean())    
    print("std :", tr.std(axis = 0))
    print("autocorrelation :", autocorr(tr))
    print("correlation with GDP :", np.corrcoef(tr, logGDP.T))
    
plt.legend() 
plt.plot(range(tr.size), logGDP, 'k-', label = "original")
plt.legend()    
plt.title('GDP')
plt.xlabel('time')
plt.show()
    

## Consumption ##
for Lam in Lambda:
    cy, tr = sm.tsa.filters.hpfilter(logConsumption, Lam)
    freq, cyper = sig.periodogram(Consumptioncy, axis=0)
    
    nobs = cyper.size
    gain = np.zeros(nobs)
    for i in range(0, nobs):
        gain[i] = cyper[i] / Consumptionper[i,0]

    plt.plot(range(tr.size), tr, label = Lam)
    print("\n\n *****", Lam, "\n\n") 
    print("mean :", tr.mean())    
    print("std :", tr.std(axis = 0))
    print("autocorrelation :", autocorr(tr))
    print("correlation with GDP :", np.corrcoef(tr, logGDP.T))
    
plt.legend() 
plt.plot(range(tr.size), logConsumption, 'k-', label = "original")
plt.legend()    
plt.title('Consumption')
plt.xlabel('time')
plt.show()

## Investment ##
for Lam in Lambda:
    cy, tr = sm.tsa.filters.hpfilter(logInvestment, Lam)
    freq, cyper = sig.periodogram(Investmentcy, axis=0)
    
    nobs = cyper.size
    gain = np.zeros(nobs)
    for i in range(0, nobs):
        gain[i] = cyper[i] / Investmentper[i,0]

    plt.plot(range(tr.size), tr, label = Lam)
    print("\n\n *****", Lam, "\n\n") 
    print("mean :", tr.mean())    
    print("std :", tr.std(axis = 0))
    print("autocorrelation :", autocorr(tr))
    print("correlation with GDP :", np.corrcoef(tr, logGDP.T))
    
plt.legend() 
plt.plot(range(tr.size), logInvestment, 'k-', label = "original")
plt.legend()    
plt.title('Investment')
plt.xlabel('time')
plt.show()


## CPI ##
for Lam in Lambda:
    cy, tr = sm.tsa.filters.hpfilter(logCPI, Lam)
    freq, cyper = sig.periodogram(CPIcy, axis=0)
    
    nobs = cyper.size
    gain = np.zeros(nobs)
    for i in range(0, nobs):
        gain[i] = cyper[i] / CPIper[i,0]

    plt.plot(range(tr.size), tr, label = Lam)
    print("\n\n *****", Lam, "\n\n") 
    print("mean :", tr.mean())    
    print("std :", tr.std(axis = 0))
    print("autocorrelation :", autocorr(tr))
    print("correlation with GDP :", np.corrcoef(tr, logGDP.T))
    
plt.legend() 
plt.plot(range(tr.size), logCPI, 'k-', label = "original")
plt.legend()    
plt.title('CPI')
plt.xlabel('time')
plt.show()


""" 
It seems that each lambda does not have big difference each other.
Also, other preperties match the known facts in the real world.
'''
