# %%
import sympy as sp
import numpy as np
from matplotlib import pyplot as plt

# Define the symbolic variables
theta, Delta, a = sp.symbols('theta Delta a')

# Define the function psi(theta)
psi = -Delta/(2*a)*sp.cot(a*sp.pi)*sp.cos(theta) + Delta/(2*a)*sp.cos(a*sp.pi)*sp.cos(a*theta - a*sp.pi)

# Compute the derivative of psi with respect to theta
psi_prime = sp.diff(psi, theta)

# Simplify the derivative expression
psi_prime_simplified = psi_prime.simplify()

# %%
Delta_val = 0.1
a_val = 0.916

# Convert the symbolic expressions to numerical functions
psi_func = sp.lambdify((theta, Delta, a), psi, "numpy")
psi_prime_func = sp.lambdify((theta, Delta, a), psi_prime_simplified, "numpy")

# Generate theta values and compute psi and psi_prime values
theta_vals = np.linspace(0, 2*np.pi, 1000)
psi_vals = psi_func(theta_vals, Delta_val, a_val)
psi_prime_vals = psi_prime_func(theta_vals, Delta_val, a_val)

# %%
plt.plot(theta_vals, psi_vals, label='psi')
plt.plot(theta_vals, psi_prime_vals, label='psi_prime')
plt.legend()
plt.show()

# %%
# get numerical values for v_theta
v_theta_vals = np.sqrt(1 - psi_prime_vals**2) / np.cos(psi_vals) - 1

# %%
plt.plot(theta_vals, v_theta_vals)
