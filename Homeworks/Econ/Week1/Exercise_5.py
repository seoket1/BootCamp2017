import scipy.optimize as opt
import numpy as np

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

def get_rvec(kvec, L, T, alpha, A, delta):
    rvec= get_rt(L, kvec, alpha, A, delta)
    return rvec
 
def get_wvec(kvec, L, T, alpha, A, delta):
    wvec = get_wt(L, kvec, alpha, A, delta)
    return wvec

def get_EulErrs_TPI_Special_Case(b32, *args):
    b21_initial_period, beta, sigma, alpha, A, delta, nvec, rvec, wvec = args
    b3 = b32
    b2 = b21_initial_period
    K = b2 + b3
    L = np.sum(nvec)
    rvec = get_rvec(kvec, L, T, alpha, A, delta)
    wvec = get_wvec(kvec, L, T, alpha, A, delta)    
    r1 = rvec[0]
    r2 = rvec[1]
    w1 = wvec[0]
    w2 = wvec[1]
    c2 = get_ct(b2, b3, r1, w1, 1)
    c3 = get_ct(b3, 0, r2, w2, 0.2)
    MU_c2 = MU_stitch(c2, sigma)
    MU_c3 = MU_stitch(c3, sigma)
    err1 = MU_c2 - beta * (1 + r2) * MU_c3
    err =  err1
    return err

def get_EulErrs_TPI(bvec, *args):
    b2, b3 = bvec
    K = b2 + b3
    beta, sigma, alpha, A, delta, nvec, rvec_22, wvec_22 = args
    L = np.sum(nvec) 
    r2 = rvec_22[1]
    r3 = rvec_22[2]
    w1 = wvec_22[0]
    w2 = wvec_22[1]
    w3 = wvec_22[2]
    c1 = get_ct(0.0, b2, 0.0, w1, 1)
    c2 = get_ct(b2, b3, r2, w2, 1)
    c3 = get_ct(b3, 0.0, r3, w3, 0.2)
    MU_c1 = MU_stitch(c1, sigma)
    MU_c2 = MU_stitch(c2, sigma)
    MU_c3 = MU_stitch(c3, sigma)
    err1 = MU_c1 - beta * (1 + r2) * MU_c2
    err2 = MU_c2 - beta * (1 + r3) * MU_c3
    err_vec = np.array([err1, err2])
    return err_vec


'''TPI parameters'''
T = 40 #int(round(6*S))
TPI_solve = True
TPI_tol = 1e-9
maxiter_TPI = 200
mindist_TPI = 1e-13
xi = 0.5
TPI_graph = True

#Overall parameters
EulDiff = False

'''getting K Vectors'''               
K_bar = K
b_initial_period = np.array([0.8*b_2, 1.1*b_3])
kvec = np.linspace(np.sum(b_initial_period), K_bar, T) 
b_args = (beta, sigma, alpha, A, delta, nvec)

#For iteration numbering
iteration_number = 1

print("\n\n---- Iteration ----\n")  

while (TPI_solve == True):
    rvec = get_rt(L, kvec, alpha, A, delta)
    wvec = get_wt(L, kvec, alpha, A, delta)
    
    #For Special Case. Basically getting b32 in period 2.
    b21_initial_period = b_initial_period[0]
    b32_initial_value = 0.05
    b_args_tpi_special_case = (b21_initial_period, beta, sigma, alpha, A, delta, nvec, rvec, wvec)
    b_result_tpi_speicial_case = opt.root(get_EulErrs_TPI_Special_Case, b32_initial_value, args = b_args_tpi_special_case)
    b32 = b_result_tpi_speicial_case.x[0]
    
    #Setting up b2t and b3t
    b2t = np.zeros(T)
    b3t = np.zeros(T)

    #For Period 1
    b2t[0] = b21_initial_period
    b3t[0] = b_initial_period[1]
    
    #For Period 2
    b3t[1] = b32

    #For K_i_prime
    K_i_prime = np.zeros(T)
    
    #To assign values to b2t and b3t
    for t in range(T): 
        b_init_tpi = np.array([0.02, 0.05])
        
        if t > 15:
            rvec_22 = np.array([rvec[t], rvec[t], rvec[t]])
            wvec_22 = np.array([wvec[t], wvec[t], wvec[t]])
        else:
            rvec_22 = np.array([rvec[t], rvec[t+1], rvec[t+2]])
            wvec_22 = np.array([wvec[t], wvec[t+1], wvec[t+2]])
            
        b_args_tpi = (beta, sigma, alpha, A, delta, nvec, rvec_22, wvec_22)
        b_result_tpi = opt.root(get_EulErrs_TPI, b_init_tpi, args = b_args_tpi)
        
        if t < (T - 1):
            b2t[t + 1] = b_result_tpi.x[0]
        if t < (T - 2):
            b3t[t + 2] = b_result_tpi.x[1]
        K_i_prime = b2t + b3t

    norm = 0
    for i in range(T):        
        temp = ((K_i_prime[i] - kvec[i])/(kvec[i]))**2
        norm = norm + temp
  
    print("Iteration: ", iteration_number, " --- Distance of Norm = ", norm)
    iteration_number = iteration_number + 1

    if norm < TPI_tol:
        TPI_solve = False
        print (b_result_tpi)
    else:
        kvec = xi * K_i_prime + (1 - xi) * kvec

print(kvec)



import matplotlib.pyplot as plt
plt.plot(kvec)

for i in range (1, T):
    if abs(kvec[i] - K_bar) <= 0.0001:
        print(i)
        print(kvec[i])


'''
Exercise 5.3
Please see the codes

Exercise 5.4
Just in six period, it went within difference of 0.0001.
'''


