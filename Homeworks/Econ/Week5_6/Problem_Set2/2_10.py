import scipy.optimize as opt
import numpy as np
from LinApp_Deriv import LinApp_Deriv
from LinApp_FindSS import LinApp_FindSS
from LinApp_Solve import LinApp_Solve
import matplotlib.pyplot as plt

def get_rt(L, K, alpha, A, delta):
    rt = alpha * A * (L/K)**(1-alpha) - delta
    return rt
 
def get_wt(L, K, alpha, A, delta):
    wt = (1-alpha) * A * (K/L)**(alpha)
    return wt

def get_ct(bt, btp1, rt, wt, ls):
    ct = (1 + rt) * bt + (ls * wt) - btp1
    return ct

def MU_stitch(ct, sigma):
    '''
    epsilon = 0.0001
    c_cnstr = ct < epsilon
    if c_cnstr:
        b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
        b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
        MU_c = 2 * b2 * ct + b1
    else:
    '''
    MU_c = ct ** (-sigma)
    return MU_c

def get_EulErrs(bvec, *args):
    b2, b3 = bvec
    K = b2 + b3
    beta, sigma, alpha, A, delta, nvec = args
    L = np.sum(nvec)
    r = get_rt(L, K, alpha, A, delta)
    w = get_wt(L, K, alpha, A, delta)
    c1 = get_ct(0.0, b2, 0.0, w, 1)
    c2 = get_ct(b2, b3, r, w, 1)
    c3 = get_ct(b3, 0.0, r, w, 0.2)
    MU_c1 = MU_stitch(c1, sigma)
    MU_c2 = MU_stitch(c2, sigma)
    MU_c3 = MU_stitch(c3, sigma)
    err1 = MU_c1 - beta * (1 + r) * MU_c2
    err2 = MU_c2 - beta * (1 + r) * MU_c3
    err_vec = np.array([err1, err2])
    return err_vec


'''Household parameters'''
S = int(3)
beta_annual = 0.96
beta = beta_annual **20
sigma = 3
nvec = np.array([1.0,1.0,0.2])
L=np.sum(nvec) # Market Clearing

'''Firm parameters'''
A = 1
alpha = 0.35
delta_annual = 0.05
delta = 1- ((1-delta_annual)**20)
# Make initial guess for solution of savings values. Note that these
# two guesses must be feasible and not violate c_t > 0 for all t
b2_init = 0.02
b3_init = 0.02
b_init = np.array([b2_init, b3_init])
b_args = (beta, sigma, alpha, A, delta, nvec)
b_result = opt.root(get_EulErrs, b_init, args=(b_args))
print(b_result)
print('Roots: ', b_result.x)

'''Steady State values'''
K = sum(b_result.x)

r = get_rt(L, K, alpha, A, delta)
w = get_wt(L, K, alpha, A, delta)

b_1 = 0
b_2 = b_result.x[0]
b_3 = b_result.x[1]

C_t1 = get_ct(0.0, b_2, 0.0, w, 1)
C_t2 = get_ct(b_2, b_3, r, w, 1)
C_t3 = get_ct(b_3, 0.0, r, w, 0.2)

b_1 = 0
b_2 = b_result.x[0]
b_3 = b_result.x[1]

print("\n\n\n---- Steady State ----\n")
print("C_t1 = ", C_t1)
print("C_t2 = ", C_t2)
print("C_t3 = ", C_t3)

print("b_1 = ", b_1)
print("b_2 = ", b_2)
print("b_3 = ", b_3)      
      
print("L = ", L)
print("K = ", K)

print("r = ", r)
print("w = ", w)



'''
Exercise 5.1

If beta_annual = 0.96

---- Steady State ----

C_t1 =  0.182412558356
C_t2 =  0.209614907072
C_t3 =  0.240873817365
b_1 =  0
b_2 =  0.0193127352391
b_3 =  0.0584115908789
L =  2.2
K =  0.077724326118
r =  2.43303025357
w =  0.201725293595

Exercise 5.2

If beta = 0.55
---- Steady State ----
C_t1 =  0.195975352642
C_t2 =  0.228615593799
C_t3 =  0.266692158088
b_1 =  0
b_2 =  0.028176959268
b_3 =  0.0768655662396
L =  2.2
K =  0.105042525508
r =  1.88635999915
w =  0.22415231191


If beta_annual = 0.99

---- Steady State ----

C_t1 =  0.214377963151
C_t2 =  0.260464851535
C_t3 =  0.316459480667
b_1 =  0
b_2 =  0.0520246493227
b_3 =  0.120018473631
L =  2.2
K =  0.172043122954
r =  1.19282040681
w =  0.266402612474


1) Every C_t increase.
2) b_2 and b_3 also increase.
3) w also goes up.
4) r decreases.
    
As all household become more patient (high beta),
fore sure, first direct effect is to save more for the future.
When people save more, Capital increase, and this affects to total output.
Therefore it also promotes consumption at each period in the long run at the fixed labor.
real interest rate decreases because capital increases.
Real interest rate is kind of the price for capital, and when capital increases,
then it means that it is not any more scarce compared to before.
So, it should be decreased.

Caveat: Actually, it depends on parameters. This function is well-desigend.
        For example, inada condition satisfies.
'''
def get_rt(L, K, zt, alpha, A, delta):
    rt = alpha * np.exp(zt) * A * (L/K)**(1-alpha) - delta
    return rt
 
def get_wt(L, K, zt, alpha, A, delta):
    wt = (1-alpha) * np.exp(zt) * A * (K/L)**(alpha)
    return wt

def get_ct(bt, btp1, rt, wt, ls):
    ct = (1 + rt) * bt + (ls * wt) - btp1
    return ct

fixed_params = (beta, sigma, alpha, A, delta, nvec)
def get_EulErrs_TPI_Linearized(bvec, args=fixed_params):
    b2pp, b3pp, b2p, b3p, b2, b3, zp, z = bvec
    beta, sigma, alpha, A, delta, nvec = args
    
    K = b2 + b3
    Kp = b2p + b3p
    #Kpp = b2pp + b3pp
    L = np.sum(nvec)
    
    r = get_rt(L, K, z, alpha, A, delta)
    rp = get_rt(L, Kp, zp, alpha, A, delta)
    #rpp = get_rt(L, Kpp, zpp, alpha, A, delta)
    
    w = get_wt(L, K, z, alpha, A, delta)
    wp = get_wt(L, Kp, zp, alpha, A, delta)
    #wpp = get_wt(L, Kpp, zpp, alpha, A, delta)
    
    c1 = get_ct(0.0, b2p, 0.0, w, 1)
    c2p = get_ct(b2p, b3pp, rp, wp, 1)
    #c3pp = get_ct(b3pp, 0.0, rpp, wpp, 0.2)
    
    c2 = get_ct(b2, b3p, r, w, 1)
    c3p = get_ct(b3p, 0.0, rp, wp, 0.2)
    
    MU_c1 = MU_stitch(c1, sigma)
    MU_c2p = MU_stitch(c2p, sigma)
    #MU_c3pp = MU_stitch(c3pp, sigma)
    
    MU_c2 = MU_stitch(c2, sigma)
    MU_c3p = MU_stitch(c3p, sigma)
    
    err1 = MU_c1 - beta * (1 + rp) * MU_c2p
    err2 = MU_c2 - beta * (1 + rp) * MU_c3p
    err_vec = np.array([err1, err2])
    return err_vec

rhoz = 0.9**20
sigmaz = 0.02
zbar = 0
 
fixed_params = (beta, sigma, alpha, A, delta, nvec)
nx,ny,nz = np.array([2, 0, 1])
theta0 = np.array([b_2, b_3, b_2, b_3, b_2, b_3, zbar, zbar])
[AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, WW, TT] = LinApp_Deriv(get_EulErrs_TPI_Linearized, fixed_params, theta0, nx, ny, nz, logX =True)

TT[1] = 0
NN = rhoz
PP, QQ, UU, RR, SS, VVV = LinApp_Solve(AA,BB,CC,DD,FF,GG,HH,JJ,KK,LL,MM,WW,TT,NN,zbar,Sylv=0)

print("PP", PP)
print("QQ", QQ)

# Linearized TPI
Xbar =([b_2, b_3])
Xinit = ([0.8*b_2, 1.1*b_3])

periods = 41
epsilon = np.zeros(periods)
Z_til = np.zeros(periods)
Z_til[0] = 0
X_til = np.zeros((2, periods))

X_til[:, 0] = X_til[:, 0] = np.log(Xinit) - np.log(Xbar)

for t in range(periods-1):
    Z_til[t+1] = rhoz * Z_til[t] + epsilon[t]
    X_til[:, t+1] = np.dot(PP, X_til[:,t]) + (QQ * Z_til[t]).T
    
# converting back to actual values
Xt = (Xbar*np.exp(X_til).T).T
zt = Z_til + zbar # zbar = 0

k2 = Xt[0, :]
k3 = Xt[1, :]

KK = k2 + k3
plt.plot(KK)
plt.title("TPI") 
plt.show()



