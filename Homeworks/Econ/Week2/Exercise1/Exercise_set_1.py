import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals, solve

# Excercise 1
A = [[0.6, 0.1, -0.3],
     [0.5, -0.4, 0.2],
     [1.0, -0.2, 1.1]]

b = [[12],
     [10],
     [-1]]

from scipy.linalg import eigvals, solve
evs = eigvals(A)
ρ = max(abs(evs))
print(ρ)

# Answer
# 1) Successive approximation
def Ax_b(A, x, b):
    return np.dot(A, x) + b

x_1 = [[1], 
     [1], 
     [1]]

while np.linalg.norm(abs(Ax_b(A, x_1, b) - x_1)) > 1e-7:
    x_1 = Ax_b(A, x_1, b)

print("\nx_1 = ", x_1)
print("\n A*x_1 + b = ", np.dot(A, x_1) + b)

# 2) Matrix algebra

I = np.eye(3) 
A_minus_I = A - I

b = np.array(b)
minus_b = b * (-1)
x_2 = np.linalg.solve(A_minus_I, minus_b)

print("\nx_2 = ", x_2)
print("\n A*x_2 + b = ", np.dot(A, x_2) + b)

# Excercise 2

# Excercise 3
beta = 0.96
wage = np.array([0.5, 1.0, 1.5])
prob = np.array([0.2, 0.4, 0.4])
c_vals = np.linspace(1, 2, 100)

w_bar = 0.1 # initial value for reservation wage

# Strategy for w bar
def strategy_for_w_bar(comp, beta, wage, prob, w_bar):
    max_term = 0
    for m in range(len(wage)):
        max_term += max(wage[m], w_bar) * prob[m]
    optimal_strategy = comp * (1 - beta) + beta * max_term
    return optimal_strategy

# Successive approxmation
def finding_w_bar(comp, beta, wage, prob, w_bar):
    while np.linalg.norm(abs(strategy_for_w_bar(comp, beta, wage, prob, w_bar) - w_bar)) > 1e-7:
            w_bar = strategy_for_w_bar(comp, beta, wage, prob, w_bar)
    return w_bar

# Store reservation wage scheme 
Reservation_wage_scheme = []
for k in range(len(c_vals)):
    temp = finding_w_bar(c_vals[k], beta, wage, prob, w_bar)
    Reservation_wage_scheme.append(temp)

# Plot the graph
print("\n\n\n")
plt.xlabel("Compensation")
plt.ylabel("Revervation Wage Scheme")
plt.title("Reservation wage Change due to Compensatioin Change")
plt.plot(c_vals, Reservation_wage_scheme)






